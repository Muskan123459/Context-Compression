"""
Layer 1 — generalised, hybrid-schema structured memory.

`GenericState` is a thin wrapper around a plain dict whose top-level keys are
fixed (so the extractor, pivot handler and budget-facts adapter can reason
about them) but whose *values* are free-form.  Any "weird" observation the
extractor wants to keep lands in the `misc` bucket without needing a schema
change.

The six top-level keys are:

    active_goals          list[str]         — user's current high-level goals
    financial_constraints dict[str, Any]    — numeric constraints (total_budget, spent, currency …)
    user_preferences      list[str]         — soft wants
    user_restrictions     list[str]         — hard limits (allergies, deal-breakers, fears …)
    locked_events         list[str]         — committed bookings / confirmed items
    misc                  dict[str, Any]    — catch-all for anything schema-less

Everything else — token counting, pinned-block rendering, the deterministic
budget-facts adapter — reads from this shape.  None of the hard-coded travel
vocabulary that lived here in v2 survives into this module.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable

from config import GLOBAL_STATE_TOKEN_CAP


# ---------------------------------------------------------------------------
# Schema constants (shared with the extractor)
# ---------------------------------------------------------------------------
LIST_FIELDS: tuple[str, ...] = (
    "active_goals",
    "user_preferences",
    "user_restrictions",
    "locked_events",
)
DICT_FIELDS: tuple[str, ...] = ("financial_constraints", "misc")
ALL_FIELDS: tuple[str, ...] = LIST_FIELDS + DICT_FIELDS

# Per-field caps keep the pinned block bounded even if the extractor is noisy.
FIELD_CAPS: dict[str, int] = {
    "active_goals":      5,
    "user_preferences":  5,
    "user_restrictions": 5,
    "locked_events":     10,
}
MAX_ITEM_CHARS: int = 140


def _default_state() -> dict[str, Any]:
    return {
        "active_goals":          [],
        "financial_constraints": {},
        "user_preferences":      [],
        "user_restrictions":     [],
        "locked_events":         [],
        "misc":                  {},
    }


# ---------------------------------------------------------------------------
# Token counting (tiktoken preferred)
# ---------------------------------------------------------------------------
def count_text_tokens(text: str) -> int:
    """Public token estimate for pinned blocks (tiktoken when available)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# GenericState
# ---------------------------------------------------------------------------
class GenericState:
    """Hybrid-schema structured memory.  Behaves like a dict plus helpers."""

    __slots__ = (
        "_state",
        "_catalog",
        "turn_count",
        "thread_id",
        "compaction_count",
    )

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
    ) -> None:
        self._state: dict[str, Any] = _default_state()
        # Sidecar catalog for structured facts harvested from tool outputs
        # (e.g. {"park hyatt tokyo": {"display_name": "Park Hyatt Tokyo",
        #                              "price_per_night_usd": 720.0,
        #                              "city": "tokyo", "kind": "hotel"}}).
        # NOT part of the schema → not rendered to the pinned prompt block,
        # but consulted by the regex safety-net when a booking message names
        # an item without a $-amount.
        self._catalog: dict[str, dict[str, Any]] = {}
        if initial:
            self.apply_ops(initial, None)
        self.turn_count: int = 0
        # Short stable identifier used by the L3 compactor when writing
        # conversation archives. Generated lazily so tests / dry-runs don't
        # depend on uuid import order.
        import uuid
        self.thread_id: str = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        self.compaction_count: int = 0

    # ── catalog (structured facts harvested from tool results) ─────────────
    def catalog_add(self, entries: Iterable[dict[str, Any]]) -> int:
        """Stash normalized entries like {name, price_per_night_usd, kind, city}.

        Returns the count of new/updated entries. No-ops on empty / invalid
        input. Keys are lowercased & whitespace-normalized for robust lookup.
        """
        n = 0
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("display_name") or "").strip()
            if not name:
                continue
            price = entry.get("price_per_night_usd")
            if price is None:
                price = entry.get("price_usd")
            if price is None:
                continue
            try:
                price_f = float(price)
            except (TypeError, ValueError):
                continue
            key = re.sub(r"\s+", " ", name.lower()).strip()
            self._catalog[key] = {
                "display_name": name,
                "price_per_night_usd": price_f,
                "kind": str(entry.get("kind") or "hotel"),
                "city": str(entry.get("city") or "").lower(),
            }
            n += 1
        return n

    # ── dict-like access (used by the budget-facts adapter and UI) ─────────
    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __contains__(self, key: str) -> bool:
        return key in self._state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    # ── predicates / exports ───────────────────────────────────────────────
    def is_empty(self) -> bool:
        return not any(self._state[f] for f in ALL_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        """Deep copy suitable for the Gradio JSON viewer / logs."""
        d = copy.deepcopy(self._state)
        fc = d.get("financial_constraints") or {}
        total = fc.get("total_budget")
        spent = fc.get("spent", 0.0)
        if isinstance(total, (int, float)):
            try:
                fc["remaining"] = round(float(total) - float(spent or 0), 2)
            except (TypeError, ValueError):
                pass
        if self._catalog:
            d["_catalog"] = copy.deepcopy(self._catalog)
        return d

    def prompt_block_token_count(self) -> int:
        p = self.to_prompt_block()
        return count_text_tokens(p) if p else 0

    # ── cross-session persistence (ChatGPT-style memory) ───────────────────
    def save(self, path: str | Path) -> None:
        """Write the 6-slot state + catalog to disk. Per-chat counters are skipped."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"state": self._state, "catalog": self._catalog}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GenericState":
        """Load state from disk; returns a fresh instance if the file is absent/bad."""
        inst = cls()
        p = Path(path)
        if not p.exists():
            return inst
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            loaded = data.get("state") or {}
            for k in ALL_FIELDS:
                if k in loaded:
                    inst._state[k] = loaded[k]
            inst._catalog = dict(data.get("catalog") or {})
        except (OSError, json.JSONDecodeError):
            pass
        return inst

    # ── mutators ───────────────────────────────────────────────────────────
    def apply_ops(
        self,
        adds: dict[str, Any] | None,
        removes: dict[str, Any] | None,
    ) -> None:
        """Merge ``adds`` into state and apply ``removes``.

        Unknown top-level keys are silently dropped (defence against a hallucinating
        extractor).  List fields dedupe case-insensitively; dict fields merge.
        Removes are fuzzy substring matches on list items and explicit key removal
        on dict fields (list of keys or dict-keyed nulls both supported).
        """
        if adds:
            for key, value in adds.items():
                if key not in ALL_FIELDS:
                    continue
                if key in LIST_FIELDS:
                    if not isinstance(value, list):
                        value = [value]
                    self._list_add(key, [str(v) for v in value if v is not None])
                else:  # dict field
                    if not isinstance(value, dict):
                        continue
                    self._dict_merge(key, value)

        if removes:
            for key, value in removes.items():
                if key not in ALL_FIELDS:
                    continue
                if key in LIST_FIELDS:
                    fragments: list[str] = []
                    if isinstance(value, list):
                        fragments = [str(v) for v in value if v]
                    elif isinstance(value, str):
                        fragments = [value]
                    self._list_remove(key, fragments)
                else:  # dict field
                    self._dict_remove(key, value)

    def drop_items_matching(self, fragments: Iterable[str]) -> int:
        """Pivot helper: drop any list-field item whose text contains one of *fragments*.

        Returns the number of items removed. Intended for cross-field purges when
        the user cancels a topic (e.g. "scratch Bali" should also clear any
        Bali-tagged goals / preferences / locked events).
        """
        frags = [f.strip().lower() for f in fragments if f and f.strip()]
        if not frags:
            return 0
        removed = 0
        for field in LIST_FIELDS:
            kept: list[str] = []
            for item in self._state[field]:
                low = item.lower()
                if any(f in low for f in frags):
                    removed += 1
                    continue
                kept.append(item)
            self._state[field] = kept
        # Also purge matching keys from misc (not from financial_constraints —
        # budget numbers outlive a pivot).
        misc = self._state.get("misc") or {}
        for k in list(misc.keys()):
            if any(f in k.lower() for f in frags) or any(
                f in str(misc[k]).lower() for f in frags
            ):
                del misc[k]
                removed += 1
        return removed

    # ── pinned-prompt rendering ────────────────────────────────────────────
    def to_prompt_block(self) -> str:
        """Render the sticky block pinned into the system prompt each turn.

        Trims from the oldest item of each list field until the block fits under
        `GLOBAL_STATE_TOKEN_CAP`.  Financial constraints are never trimmed —
        numbers are the one thing SmolLM3 cannot recover on its own.
        """
        if self.is_empty():
            return ""

        scratch: dict[str, Any] = {
            "active_goals":          list(self._state["active_goals"]),
            "user_preferences":      list(self._state["user_preferences"]),
            "user_restrictions":     list(self._state["user_restrictions"]),
            "locked_events":         list(self._state["locked_events"]),
            "financial_constraints": dict(self._state["financial_constraints"]),
            "misc":                  dict(self._state["misc"]),
        }

        # Drop order: trim misc first, then locked_events, preferences,
        # goals, restrictions (restrictions are safety-critical so we keep them
        # longest — allergies, fears, etc.).
        trim_order = ("misc", "locked_events", "user_preferences",
                      "active_goals", "user_restrictions")

        while True:
            text = self._render(scratch)
            if count_text_tokens(text) <= GLOBAL_STATE_TOKEN_CAP:
                return text
            trimmed = False
            for field in trim_order:
                bucket = scratch[field]
                if isinstance(bucket, list) and len(bucket) > 0:
                    bucket.pop(0)
                    trimmed = True
                    break
                if isinstance(bucket, dict) and bucket:
                    bucket.pop(next(iter(bucket)))
                    trimmed = True
                    break
            if not trimmed:
                return text

    # ── internals ──────────────────────────────────────────────────────────
    @staticmethod
    def _clip(s: str) -> str:
        s = s.strip()
        if len(s) > MAX_ITEM_CHARS:
            s = s[: MAX_ITEM_CHARS - 1] + "…"
        return s

    def _list_add(self, field: str, items: list[str]) -> None:
        bucket = self._state[field]
        seen = {x.lower() for x in bucket}
        for raw in items:
            x = self._clip(raw)
            if not x:
                continue
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            bucket.append(x)
        cap = FIELD_CAPS.get(field, 10)
        while len(bucket) > cap:
            bucket.pop(0)

    def _list_remove(self, field: str, fragments: list[str]) -> None:
        if not fragments:
            return
        frags = [f.strip().lower() for f in fragments if f and f.strip()]
        if not frags:
            return
        bucket = self._state[field]
        kept: list[str] = []
        for item in bucket:
            low = item.lower()
            if any(f in low for f in frags):
                continue
            kept.append(item)
        self._state[field] = kept

    def _dict_merge(self, field: str, patch: dict[str, Any]) -> None:
        current: dict[str, Any] = self._state[field]
        for k, v in patch.items():
            if v is None:
                current.pop(k, None)
                continue
            # Coerce numeric strings so downstream math adapters don't have to
            # second-guess the model.
            if isinstance(v, str):
                maybe_num = _try_number(v)
                if maybe_num is not None:
                    v = maybe_num
            current[k] = v

    def _dict_remove(self, field: str, value: Any) -> None:
        current: dict[str, Any] = self._state[field]
        if isinstance(value, list):
            for k in value:
                current.pop(str(k), None)
        elif isinstance(value, dict):
            for k in value:
                current.pop(str(k), None)
        elif isinstance(value, str):
            current.pop(value, None)

    def _render(self, src: dict[str, Any]) -> str:
        lines: list[str] = ["## Current state (ALWAYS respect this)"]

        if src["active_goals"]:
            lines.append("Active goals:")
            for g in src["active_goals"]:
                lines.append(f"  • {self._clip(g)}")

        fc = src["financial_constraints"]
        if fc:
            lines.append("Financial constraints:")
            total = fc.get("total_budget")
            spent = fc.get("spent", 0)
            currency = fc.get("currency", "USD")
            if isinstance(total, (int, float)) or isinstance(spent, (int, float)):
                total_f = float(total) if isinstance(total, (int, float)) else None
                spent_f = float(spent) if isinstance(spent, (int, float)) else 0.0
                if total_f is not None:
                    remaining = round(total_f - spent_f, 2)
                    lines.append(
                        f"  • Budget: {currency} {total_f:,.0f} total | "
                        f"spent {currency} {spent_f:,.0f} | "
                        f"remaining {currency} {remaining:,.0f}"
                    )
                elif spent_f:
                    lines.append(
                        f"  • Budget: spent {currency} {spent_f:,.0f} (total not stated)"
                    )
            for k, v in fc.items():
                if k in {"total_budget", "spent", "currency"}:
                    continue
                lines.append(f"  • {k}: {self._clip(str(v))}")

        if src["user_preferences"]:
            lines.append("User preferences:")
            for p in src["user_preferences"]:
                lines.append(f"  • {self._clip(p)}")

        if src["user_restrictions"]:
            lines.append("User restrictions (hard limits — treat as non-negotiable):")
            for r in src["user_restrictions"]:
                lines.append(f"  • {self._clip(r)}")

        if src["locked_events"]:
            lines.append("Confirmed / locked:")
            for e in src["locked_events"]:
                lines.append(f"  • {self._clip(e)}")

        misc = src["misc"]
        if misc:
            lines.append("Notes:")
            for k, v in misc.items():
                lines.append(f"  • {k}: {self._clip(str(v))}")

        return "\n".join(lines)


def _try_number(s: str) -> float | int | None:
    """Best-effort parse of '$3,800', '3800.50', '7 nights' → numeric."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Deterministic budget-facts adapter
# ---------------------------------------------------------------------------
#
# SmolLM3-3B cannot do arithmetic reliably. When the pinned state has numeric
# financial_constraints we compute the relevant sums in Python and inject a
# LOCKED block the model is forbidden to modify. The adapter is opt-in by
# schema: if no financial_constraints fields are present, nothing is emitted,
# so the compression engine stays domain-free for non-travel use cases.
# ---------------------------------------------------------------------------
_BUDGET_TRIGGER_RE = re.compile(
    r"\b(budget|afford|affordable|remaining|left|spent|cost|price|total|within|over\s+budget)\b",
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_NIGHTS_RE = re.compile(r"(\d+)\s*nights?", re.IGNORECASE)


def build_budget_facts_block(state: GenericState, user_msg: str) -> str:
    """Pre-computed financial facts pinned in front of tool results.

    Reads `state["financial_constraints"]` and only fires when there are
    numeric budget values to anchor against.  Identical output to the v2 block,
    but sourced from the new dict-backed state instead of dataclass fields.
    """
    fc = state.get("financial_constraints") or {}
    total = fc.get("total_budget")
    spent = fc.get("spent", 0.0)
    currency = fc.get("currency", "USD")

    try:
        total_f: float | None = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    try:
        spent_f: float = float(spent or 0.0)
    except (TypeError, ValueError):
        spent_f = 0.0

    if total_f is None and spent_f == 0.0:
        return ""

    remaining = round((total_f or 0.0) - spent_f, 2) if total_f is not None else -spent_f

    amounts = [float(m.group(1).replace(",", "")) for m in _DOLLAR_RE.finditer(user_msg)]
    nights_hint = _NIGHTS_RE.search(user_msg)
    nights = int(nights_hint.group(1)) if nights_hint else None
    keyword_hit = bool(_BUDGET_TRIGGER_RE.search(user_msg))

    lines: list[str] = [
        "────────────────────────────────────────────────────",
        "BUDGET FACTS (system-computed — LOCKED):",
        "  Rules for the assistant:",
        "   • Copy these numbers verbatim.",
        "   • Do NOT add your own 'Affordability check' or per-category sums.",
        "   • Do NOT invent flight legs, hotel counts, or trip days not listed here.",
        f"- Total budget: {currency} {total_f:,.0f}" if total_f is not None else "- Total budget: (not set)",
        f"- Spent so far: {currency} {spent_f:,.0f}",
        f"- Remaining: {currency} {remaining:,.0f}",
    ]

    if remaining > 0 and (keyword_hit or amounts):
        lines.append("- Max nightly rate you can still afford:")
        for n in (1, 2, 3, 4, 5):
            lines.append(
                f"  • {n} night{'s' if n > 1 else ''}: up to {currency} {remaining / n:,.0f}/night"
            )

    if amounts:
        lines.append("- Affordability check for amounts mentioned in the question:")
        for amt in amounts:
            if nights:
                total_cost = amt * nights
                verdict = (
                    "WITHIN BUDGET"
                    if total_cost <= remaining
                    else f"OVER BUDGET by {currency} {total_cost - remaining:,.0f}"
                )
                lines.append(
                    f"  • {currency} {amt:,.0f}/night × {nights} nights = "
                    f"{currency} {total_cost:,.0f} → {verdict} (remaining {currency} {remaining:,.0f})"
                )
            else:
                verdict = (
                    "WITHIN BUDGET"
                    if amt <= remaining
                    else f"OVER BUDGET by {currency} {amt - remaining:,.0f}"
                )
                lines.append(
                    f"  • {currency} {amt:,.0f} → {verdict} (remaining {currency} {remaining:,.0f})"
                )
    else:
        lines.append(
            "- No specific dollar amount in this turn — do NOT invent one for affordability."
        )

    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backwards-compat alias — anything still importing `GlobalState` keeps working
# ---------------------------------------------------------------------------
GlobalState = GenericState  # deprecated; prefer `GenericState`

__all__ = [
    "ALL_FIELDS",
    "DICT_FIELDS",
    "GenericState",
    "GlobalState",
    "LIST_FIELDS",
    "build_budget_facts_block",
    "count_text_tokens",
]

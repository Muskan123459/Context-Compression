"""
Layer 1 — domain-free structured-memory extractor.

Every turn the extractor:

1. Runs a tiny regex fast-path to capture the initial budget ("budget is $3,000"
   → financial_constraints.total_budget) without an LLM call.

2. Calls one LLM with the current state, the user message, and the PRIOR
   assistant reply. The LLM returns a JSON patch:
       {"add": {…}, "remove": {…}, "catalog_add": [...]}
   - `add`/`remove` mutate the 6-slot hybrid schema (goals, prefs, restrictions,
     locked_events, financial_constraints, misc).
   - Bookings are handled here: when the user confirms one, the LLM emits the
     matching `spent` delta in `financial_constraints` and the summary in
     `locked_events`.
   - `catalog_add` adds named items with USD prices that appeared in the prior
     assistant reply so later bookings can resolve them.

3. Pivot keywords ("scratch", "instead", …) trigger a short second LLM call
   that returns fragments to purge cross-field.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from config import MODEL, PIVOT_CONFIRM_MAX_TOKENS, STATE_EXTRACTOR_MAX_TOKENS
from global_state import ALL_FIELDS, GenericState

if TYPE_CHECKING:
    from openai import OpenAI

# ---------------------------------------------------------------------------
# Regex safety-net — numeric only, domain-agnostic
# ---------------------------------------------------------------------------
_AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")

_BUDGET_TRIGGER_RE = re.compile(r"\bbudget\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------
_PIVOT_RE = re.compile(
    r"\b("
    r"scratch(?:\s+that)?|scrap(?:\s+that)?|"
    r"instead(?:\s+of)?|change(?:\s+to|\s+that)?|"
    r"forget(?:\s+about)?|cancel|never\s*mind|nevermind|"
    r"actually(?:,|\s)|drop\s+(?:that|the)|"
    r"no\s+longer|not\s+interested\s+in|"
    r"let'?s\s+do|let\s+us\s+do"
    r")\b",
    re.IGNORECASE,
)


def _was_pivot_triggered(message: str) -> bool:
    return bool(_PIVOT_RE.search(message))


# ---------------------------------------------------------------------------
# Debug dump — write the raw extractor/pivot LLM I/O to disk so we can
# actually see what SmolLM3 emits when the JSON patch comes back empty.
# Toggle with DUMP_EXTRACTOR env var (default on).
# ---------------------------------------------------------------------------
_LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _dump_extractor_call(
    kind: str,
    turn_no: int,
    payload: dict,
    trace: list[str],
) -> None:
    if os.getenv("DUMP_EXTRACTOR", "1") == "0":
        return
    try:
        _LOGS_DIR.mkdir(exist_ok=True)
        path = _LOGS_DIR / f"extractor-turn-{turn_no:03d}.txt"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"=== {kind} — turn {turn_no} ===\n")
            for section, value in payload.items():
                fh.write(f"── {section} ──\n")
                if isinstance(value, (dict, list)):
                    fh.write(json.dumps(value, ensure_ascii=False, indent=2))
                else:
                    fh.write(str(value))
                fh.write("\n\n")
        trace.append(f"{kind} dump → logs/extractor-turn-{turn_no:03d}.txt")
    except Exception as exc:
        trace.append(f"{kind} dump skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------
_EXTRACTOR_SYSTEM = """\
You are a structured-memory updater for a travel-planning assistant.

You see three things:
  CURRENT STATE      — JSON with the 6-slot memory and a `_catalog` sidecar.
  PRIOR ASSISTANT REPLY — what was just shown to the user (may be empty on turn 1).
  USER MESSAGE       — the user's latest turn.

Return ONLY a JSON object (no prose, no code fences) of shape:
{
  "add":           { <subset of 6-slot schema — NO bookings, NO spent, NO locked_events> },
  "remove":        { <subset of 6-slot schema — NO locked_events, NO spent> },
  "catalog_add":   [ {"name": "...", "price_usd": <num>, "kind": "hotel|flight|other"}, ... ],
  "bookings":      [ {"name": "...", "unit_price_usd": <num>, "quantity": <int>, "unit": "night|ticket|trip"} ],
  "cancellations": [ {"name": "<fragment that appears in current locked_events>", "refund_usd": <num from that same locked_events string>} ]
}

Schema for add/remove (6 slots):
  active_goals          list of short strings  — high-level goals
  financial_constraints object                 — {total_budget, spent, currency, ...}
  user_preferences      list of short strings  — soft wants (cabin class, style, vibe, …)
  user_restrictions     list of short strings  — hard limits (allergies, fears, dietary)
  locked_events         list of short strings  — (Python writes these from `bookings`; do NOT write here yourself)
  misc                  object                 — free-form notes

Booking rules — YOU ONLY IDENTIFY. PYTHON DOES THE MATH.
- If the USER MESSAGE confirms a specific item (e.g. "book the ANA one",
  "confirm Park Hyatt for 3 nights", "go ahead and book it"), emit a
  `bookings` entry:
      {"name": "<canonical item name>",
       "unit_price_usd": <unit price from PRIOR REPLY or _catalog — NEVER guess>,
       "quantity":       <number of nights / tickets / trips; default 1>,
       "unit":           "night" | "ticket" | "trip"}
- Do NOT compute totals. Do NOT write `financial_constraints.spent`. Do NOT
  write `locked_events`. Python updates those from your `bookings` list.
- Offers, suggestions and comparisons are NOT bookings.
- If you cannot find a clear unit_price, skip the booking (better empty than wrong).

Cancellation rules — YOU ONLY IDENTIFY. PYTHON DOES THE MATH.
- If the user asks to cancel / remove / undo / refund prior bookings
  ("remove all the bookings", "cancel the hotel", "start fresh",
  "forget everything I booked", "I'm not going to Tokyo anymore"), emit
  `cancellations` entries — one per booking being reversed.
- `name` must be a short fragment that appears inside an existing
  `locked_events` string (e.g. "Park Hyatt Tokyo", "Air India").
- `refund_usd` must come from that same locked_events string (it already
  contains the exact total). NEVER invent. NEVER guess.
- For "remove all" / "start fresh" / topic pivots that invalidate the trip
  (e.g. "I'm going to Switzerland now, forget Tokyo"), include EVERY current
  locked_events entry with its amount.
- Python will remove matching locked_events and subtract refund_usd from
  financial_constraints.spent.
- Do NOT also put these in `remove.locked_events` — just use `cancellations`.

catalog_add rules:
- Extract named items with USD prices from the PRIOR ASSISTANT REPLY only.
- Use `kind: "hotel"` for per-night prices, `kind: "flight"` for flight fares.
- Do NOT invent prices. Skip items already in `_catalog`.

General rules:
- `add` = facts the user JUST introduced. Don't restate existing state.
- `remove` for list fields: short text fragments; for dict fields: list of keys.
- Capture soft preferences ("economy class", "mountains", "5 days") as
  `user_preferences`. Safety-critical ("allergic to shellfish") → `user_restrictions`.
- Keep each string under 12 words. JSON only.
"""

_PIVOT_SYSTEM = """\
A user just used a pivot phrase ("scratch", "instead", "forget", "actually", …).
Decide if they are cancelling a topic that was discussed earlier.

Reply JSON only:
{"pivoted": true|false, "removed_fragments": ["...short text fragments..."]}

Rules:
- removed_fragments should be strings we can substring-match against existing
  state entries (e.g. "Bali", "helicopter tour", "beach vacation").
- Under 6 fragments, each under 6 words.
- If the user is only adjusting detail (not replacing a topic), return pivoted=false.
"""


# ---------------------------------------------------------------------------
# Regex safety-net
# ---------------------------------------------------------------------------
def _amounts(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "")) for m in _AMOUNT_RE.finditer(text)]


def apply_regex_safety_net(state: GenericState, user_message: str) -> None:
    """Tiny fast-path: capture the initial total_budget from "budget is $3,000".
    Bookings and `spent` updates are now handled by the LLM in `llm_extract`.
    """
    fc = dict(state.get("financial_constraints") or {})
    if _BUDGET_TRIGGER_RE.search(user_message) and fc.get("total_budget") is None:
        amounts = _amounts(user_message)
        if amounts:
            fc["total_budget"] = max(amounts)
            fc.setdefault("currency", "USD")
            state.apply_ops({"financial_constraints": fc}, None)


# ---------------------------------------------------------------------------
# LLM extractor
# ---------------------------------------------------------------------------
def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_ops(
    raw: str,
) -> tuple[dict, dict, list[dict], list[dict], list[dict]]:
    """Parse the extractor JSON into
    (adds, removes, catalog_add, bookings, cancellations).

    Anything malformed → empty ops (never raises). Unknown keys are filtered.
    The LLM is forbidden from writing `financial_constraints.spent` or
    `locked_events` directly (in either `add` or `remove`); those come from
    the `bookings` / `cancellations` lists that Python processes.
    """
    try:
        data = json.loads(_strip_json_fence(raw))
    except Exception:
        return {}, {}, [], [], []
    if not isinstance(data, dict):
        return {}, {}, [], [], []

    def _filter(d: object) -> dict:
        if not isinstance(d, dict):
            return {}
        out = {k: v for k, v in d.items() if k in ALL_FIELDS}
        # Safety: strip LLM-computed booking fields from BOTH add and remove —
        # Python is the sole writer for locked_events and spent (via
        # `bookings` / `cancellations`).
        out.pop("locked_events", None)
        fc = out.get("financial_constraints")
        if isinstance(fc, dict):
            fc.pop("spent", None)
            if not fc:
                out.pop("financial_constraints", None)
        return out

    def _as_dict_list(v: object) -> list[dict]:
        return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []

    return (
        _filter(data.get("add")),
        _filter(data.get("remove")),
        _as_dict_list(data.get("catalog_add")),
        _as_dict_list(data.get("bookings")),
        _as_dict_list(data.get("cancellations")),
    )


def _apply_bookings(
    state: GenericState,
    bookings: list[dict],
    trace: list[str],
) -> None:
    """Python-side math for every booking the LLM identified.

    Each booking: {name, unit_price_usd, quantity, unit}. We compute
    total = unit_price × quantity, increment financial_constraints.spent,
    and append a deterministic summary string to locked_events.
    """
    if not bookings:
        return
    fc = dict(state.get("financial_constraints") or {})
    try:
        spent = float(fc.get("spent") or 0.0)
    except (TypeError, ValueError):
        spent = 0.0
    currency = fc.get("currency") or "USD"

    summaries: list[str] = []
    for b in bookings:
        name = str(b.get("name") or "").strip()
        try:
            unit_price = float(b.get("unit_price_usd"))
        except (TypeError, ValueError):
            continue
        try:
            qty = int(b.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1
        if not name or unit_price <= 0 or qty < 1:
            continue
        unit = str(b.get("unit") or "").strip().lower() or "item"
        total = round(unit_price * qty, 2)
        spent = round(spent + total, 2)
        if qty > 1:
            summaries.append(
                f"Booked {name}: {qty} {unit}{'s' if qty != 1 else ''} × "
                f"{currency} {unit_price:,.0f} = {currency} {total:,.0f}"
            )
        else:
            summaries.append(f"Booked {name}: {currency} {total:,.0f}")

    if not summaries:
        return

    state.apply_ops(
        {
            "financial_constraints": {"spent": spent},
            "locked_events": summaries,
        },
        None,
    )
    trace.append(
        f"bookings applied (Python math): +{len(summaries)} event(s), "
        f"new spent = {currency} {spent:,.2f}"
    )


def _apply_cancellations(
    state: GenericState,
    cancellations: list[dict],
    trace: list[str],
) -> None:
    """Python-side math for every cancellation the LLM identified.

    Each entry: {name, refund_usd}. We remove any locked_events string that
    contains `name` (case-insensitive substring match) and subtract
    `refund_usd` from financial_constraints.spent. Spent is floored at 0.
    """
    if not cancellations:
        return

    current_locked = list(state.get("locked_events") or [])
    fc = dict(state.get("financial_constraints") or {})
    try:
        spent = float(fc.get("spent") or 0.0)
    except (TypeError, ValueError):
        spent = 0.0
    currency = fc.get("currency") or "USD"

    remove_fragments: list[str] = []
    total_refund = 0.0
    removed_labels: list[str] = []

    for c in cancellations:
        name = str(c.get("name") or "").strip()
        try:
            refund = float(c.get("refund_usd"))
        except (TypeError, ValueError):
            continue
        if not name or refund <= 0:
            continue
        needle = name.lower()
        match = next(
            (entry for entry in current_locked if needle in entry.lower()),
            None,
        )
        if match is None:
            continue
        remove_fragments.append(match)
        current_locked.remove(match)
        total_refund = round(total_refund + refund, 2)
        removed_labels.append(name)

    if not remove_fragments:
        return

    new_spent = round(max(0.0, spent - total_refund), 2)
    state.apply_ops(
        {"financial_constraints": {"spent": new_spent}},
        {"locked_events": remove_fragments},
    )
    trace.append(
        f"cancellations applied (Python math): -{len(remove_fragments)} event(s) "
        f"({', '.join(removed_labels)}), refunded {currency} {total_refund:,.2f}, "
        f"new spent = {currency} {new_spent:,.2f}"
    )


def llm_extract(
    state: GenericState,
    user_message: str,
    prior_assistant_reply: str,
    client: "OpenAI",
    trace: list[str],
) -> None:
    """Single LLM call: updates the 6-slot state AND handles bookings +
    catalog harvesting from the prior assistant reply."""
    if not user_message or not user_message.strip():
        return

    state_before = state.to_dict()
    current = json.dumps(state_before, ensure_ascii=False)
    user_prompt = (
        f"CURRENT STATE:\n{current}\n\n"
        f"PRIOR ASSISTANT REPLY:\n{(prior_assistant_reply or '')[:2500]}\n\n"
        f"USER MESSAGE:\n{user_message[:2000]}\n\n"
        f"Return JSON only."
    )

    raw = ""
    error_str = ""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=STATE_EXTRACTOR_MAX_TOKENS,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        trace.append(f"state extractor (LLM) skipped: {error_str}")

    adds, removes, catalog_add, bookings, cancellations = (
        _parse_ops(raw) if raw else ({}, {}, [], [], [])
    )

    if adds or removes:
        state.apply_ops(adds, removes)
        if adds:
            trace.append(f"state extractor add: {json.dumps(adds, ensure_ascii=False)[:160]}")
        if removes:
            trace.append(f"state extractor remove: {json.dumps(removes, ensure_ascii=False)[:160]}")
    elif not error_str and not bookings and not catalog_add and not cancellations:
        trace.append("state extractor: no ops (empty / malformed JSON)")

    if catalog_add:
        n = state.catalog_add(catalog_add)
        if n:
            trace.append(f"catalog harvested from reply: {n} item(s)")

    _apply_bookings(state, bookings, trace)
    _apply_cancellations(state, cancellations, trace)

    _dump_extractor_call(
        "llm_extract",
        state.turn_count,
        {
            "user_message": user_message,
            "prior_assistant_reply": prior_assistant_reply,
            "state_before": state_before,
            "raw_response": raw or f"<no response — {error_str}>",
            "parsed_adds": adds,
            "parsed_removes": removes,
            "parsed_catalog_add": catalog_add,
            "parsed_bookings": bookings,
            "parsed_cancellations": cancellations,
            "state_after": state.to_dict(),
        },
        trace,
    )


# ---------------------------------------------------------------------------
# Pivot handler
# ---------------------------------------------------------------------------
def maybe_handle_pivot(
    state: GenericState,
    user_message: str,
    client: "OpenAI",
    trace: list[str],
) -> list[str]:
    """If a pivot keyword fires, confirm via LLM and purge matching items.

    Returns the list of matched fragments (possibly empty) for downstream use
    by the compactor, which will also skip them when summarising old turns.
    """
    if not _was_pivot_triggered(user_message):
        return []

    current_goals = state.get("active_goals") or []
    current_prefs = state.get("user_preferences") or []
    current_locked = state.get("locked_events") or []
    if not (current_goals or current_prefs or current_locked):
        return []

    pivot_prompt = (
        f"Existing active goals: {json.dumps(current_goals, ensure_ascii=False)}\n"
        f"Existing preferences : {json.dumps(current_prefs, ensure_ascii=False)}\n"
        f"Existing locked      : {json.dumps(current_locked, ensure_ascii=False)}\n\n"
        f"USER MESSAGE:\n{user_message[:1000]}\n\n"
        f"Return JSON only."
    )
    raw = ""
    error_str = ""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _PIVOT_SYSTEM},
                {"role": "user",   "content": pivot_prompt},
            ],
            max_tokens=PIVOT_CONFIRM_MAX_TOKENS,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        trace.append(f"pivot confirm skipped: {error_str}")

    parsed: dict = {}
    if raw:
        try:
            parsed = json.loads(_strip_json_fence(raw))
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            trace.append("pivot confirm: malformed JSON")
            parsed = {}

    fragments: list[str] = []
    if parsed.get("pivoted"):
        fragments_raw = parsed.get("removed_fragments") or []
        if isinstance(fragments_raw, list):
            fragments = [str(f).strip() for f in fragments_raw if f]

    removed_count = 0
    if fragments:
        removed_count = state.drop_items_matching(fragments)
        trace.append(
            f"pivot detected — purged {removed_count} item(s) matching "
            f"{json.dumps(fragments, ensure_ascii=False)}"
        )

    _dump_extractor_call(
        "pivot_confirm",
        state.turn_count,
        {
            "user_message": user_message,
            "existing_goals": current_goals,
            "existing_prefs": current_prefs,
            "existing_locked": current_locked,
            "raw_response": raw or f"<no response — {error_str}>",
            "parsed": parsed,
            "fragments_matched": fragments,
            "items_purged": removed_count,
        },
        trace,
    )
    return fragments


# ---------------------------------------------------------------------------
# One-call entry point: budget fast-path → LLM extract → pivot purge
# ---------------------------------------------------------------------------
def run_extractor(
    state: GenericState,
    user_message: str,
    prior_assistant_reply: str,
    client: "OpenAI",
    trace: list[str],
) -> list[str]:
    """Full L1 turn update. Returns pivot fragments (if any) for the compactor."""
    apply_regex_safety_net(state, user_message)
    llm_extract(state, user_message, prior_assistant_reply, client, trace)
    return maybe_handle_pivot(state, user_message, client, trace)


__all__ = [
    "apply_regex_safety_net",
    "llm_extract",
    "maybe_handle_pivot",
    "run_extractor",
]

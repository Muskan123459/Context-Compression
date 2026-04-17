"""
v2 — bounded GlobalState pinned into the system prompt (L1 sticky state).

Fields and caps match the build plan; `to_prompt_block()` trims until under
GLOBAL_STATE_TOKEN_CAP (tiktoken when available, else char heuristic).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config import GLOBAL_STATE_TOKEN_CAP

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
# GlobalState
# ---------------------------------------------------------------------------


@dataclass
class GlobalState:
    destinations: list[str] = field(default_factory=list)
    budget_total: float | None = None
    budget_spent: float = 0.0
    preferences: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    turn_count: int = 0

    MAX_DEST: int = 6
    MAX_PREF: int = 5
    MAX_CONS: int = 3
    MAX_DEC: int = 8
    MAX_ITEM_CHARS: int = 120

    def is_empty(self) -> bool:
        return not (
            self.destinations
            or self.budget_total is not None
            or self.budget_spent > 0
            or self.preferences
            or self.constraints
            or self.decisions
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.budget_total is not None:
            d["budget_total"] = self.budget_total
        d["budget_remaining"] = self._remaining()
        return d

    def _remaining(self) -> float | None:
        if self.budget_total is None:
            return None
        return round(self.budget_total - self.budget_spent, 2)

    def _clip_item(self, s: str) -> str:
        s = s.strip()
        if len(s) > self.MAX_ITEM_CHARS:
            s = s[: self.MAX_ITEM_CHARS - 1] + "…"
        return s

    def _dedupe_append(self, items: list[str], new_items: list[str], cap: int) -> None:
        seen = {x.lower() for x in items}
        for raw in new_items:
            x = self._clip_item(raw)
            if not x:
                continue
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(x)
        while len(items) > cap:
            items.pop(0)

    def add_destinations(self, names: list[str]) -> None:
        self._dedupe_append(self.destinations, names, self.MAX_DEST)

    def add_preferences(self, prefs: list[str]) -> None:
        self._dedupe_append(self.preferences, prefs, self.MAX_PREF)

    def add_constraints(self, cons: list[str]) -> None:
        self._dedupe_append(self.constraints, cons, self.MAX_CONS)

    def add_decisions(self, decs: list[str]) -> None:
        self._dedupe_append(self.decisions, decs, self.MAX_DEC)

    def prompt_block_token_count(self) -> int:
        p = self.to_prompt_block()
        return count_text_tokens(p) if p else 0

    def to_prompt_block(self) -> str:
        if self.is_empty():
            return ""

        dest = list(self.destinations)
        prefs = list(self.preferences)
        cons = list(self.constraints)
        decs = list(self.decisions)

        while True:
            lines = ["## Current trip state (ALWAYS respect this)"]
            if dest:
                lines.append("Destinations: " + ", ".join(self._clip_item(d) for d in dest))
            if self.budget_total is not None:
                rem = round(self.budget_total - self.budget_spent, 2)
                lines.append(
                    f"Budget: ${self.budget_total:,.0f} total | "
                    f"spent ${self.budget_spent:,.0f} | "
                    f"remaining ${rem:,.0f}"
                )
            elif self.budget_spent > 0:
                lines.append(f"Budget: spent ${self.budget_spent:,.0f} (total not stated)")
            if prefs:
                lines.append("Preferences:")
                for p in prefs:
                    lines.append(f"  • {self._clip_item(p)}")
            if cons:
                lines.append("Constraints:")
                for c in cons:
                    lines.append(f"  • {self._clip_item(c)}")
            if decs:
                lines.append("Confirmed:")
                for d in decs:
                    lines.append(f"  • {self._clip_item(d)}")
            text = "\n".join(lines)

            if count_text_tokens(text) <= GLOBAL_STATE_TOKEN_CAP or (
                not prefs and not decs and len(dest) <= 1 and not cons
            ):
                return text

            if len(prefs) > 1:
                prefs.pop(0)
            elif len(decs) > 1:
                decs.pop(0)
            elif len(dest) > 1:
                dest.pop(0)
            elif cons:
                cons.pop(0)
            else:
                return text

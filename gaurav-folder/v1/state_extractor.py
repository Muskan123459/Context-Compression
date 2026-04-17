"""
v2 — hybrid GlobalState extraction: fast regex merge + optional vLLM JSON pass.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from config import MODEL, STATE_EXTRACTOR_MAX_TOKENS
from global_state import GlobalState
from tool_router import extract_all_cities, is_booking_confirmation

if TYPE_CHECKING:
    from openai import OpenAI

_AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")

_PREF_VERB_RE = re.compile(
    r"\b(prefer|preference|preferences|want|don't want|do not want|hate|love|should|must|"
    r"need to|care about|important that|rather|would like|i'd like)\b",
    re.IGNORECASE,
)

_DIETARY_RE = re.compile(
    r"\b(allergic to|allergy to|allergy\b|allergies\b|can't eat|cannot eat|"
    r"gluten|celiac|coeliac|shellfish|peanut|tree nut|lactose|dairy-free|vegan|vegetarian|"
    r"kosher|halal|pescatarian)\b",
    re.IGNORECASE,
)

_EXTRACTOR_SYSTEM = """\
Extract ONLY new facts from the user message about travel preferences or hard constraints.
Do NOT invent. Do NOT copy tool results. If nothing applies, return {"preferences":[],"constraints":[]}.

Output a single JSON object, no markdown, no prose:
{"preferences": ["..."], "constraints": ["..."]}

Rules:
- At most 3 items per list.
- Each string at most 12 words.
- preferences = soft wants (neighborhood vibe, pricing style, pace of trip).
- constraints = hard limits (allergies, must-haves, deal-breakers).
"""


def _all_amounts(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "")) for m in _AMOUNT_RE.finditer(text)]


def merge_regex(state: GlobalState, user_message: str) -> None:
    """Update *state* with rule-based extraction (no LLM)."""
    msg = user_message
    lower = msg.lower()
    booking = is_booking_confirmation(msg)

    if not booking:
        cities = extract_all_cities(msg)
        if cities:
            lower = msg.lower()
            # Delhi is often the departure city, not a trip destination
            if re.search(r"\bfrom\s+delhi\b", lower) and not re.search(
                r"\b(in|at|near|visit|stay|staying|nights?\s+in)\s+delhi\b",
                lower,
            ):
                cities = [c for c in cities if c != "Delhi"]
            if cities:
                state.add_destinations(cities)

        if state.budget_total is None and (
            re.search(r"\bbudget\b", lower) or "total budget" in lower
        ):
            amounts = _all_amounts(msg)
            if amounts:
                state.budget_total = max(amounts)

        dm = _DIETARY_RE.search(msg)
        if dm:
            frag = dm.group(0).strip()
            state.add_constraints([f"Dietary / restriction: {frag}"])

    else:
        amounts = _all_amounts(msg)
        if amounts:
            state.budget_spent += sum(amounts)
            summary = msg.strip().split("\n")[0].strip()
            if len(summary) > 160:
                summary = summary[:157] + "…"
            state.add_decisions([summary])


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def maybe_llm_extract(
    state: GlobalState,
    user_message: str,
    client: OpenAI,
    trace: list[str],
) -> None:
    """Optional vLLM call for free-text preferences/constraints."""
    if is_booking_confirmation(user_message):
        return
    if not _PREF_VERB_RE.search(user_message):
        return

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": f"Message:\n{user_message[:2000]}\n\nExtract JSON only:",
                },
            ],
            max_tokens=STATE_EXTRACTOR_MAX_TOKENS,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = _strip_json_fence(raw)
        data = json.loads(raw)
        prefs = data.get("preferences") or []
        cons = data.get("constraints") or []
        if isinstance(prefs, list):
            state.add_preferences([str(p) for p in prefs if p])
        if isinstance(cons, list):
            state.add_constraints([str(c) for c in cons if c])
    except Exception as exc:
        trace.append(f"state extractor (LLM) skipped: {type(exc).__name__}: {exc}")

"""
v1 — Rule-based tool router.

Sits between the user message and the LLM call.
Decides which tools to pre-call, executes them, and returns a formatted
context block that gets injected into the conversation before the LLM
generates its reply.

This approach is reliable with any model regardless of its function-calling
capability: the LLM only needs to read the injected results and write a
text reply — no tool_call JSON required from the model.

The routing rules are intentionally simple keyword matches, as recommended
by the problem statement: "make the tool definitions rule based (if you want),
but the tools should return valid outputs."
"""
from __future__ import annotations

import json
import re
from typing import NamedTuple

from tools import dispatch_tool


def _has_word(text: str, *words: str) -> bool:
    """True if any of *words* appears as a word-start match in *text* (handles plurals/gerunds)."""
    for w in words:
        if re.search(r"\b" + re.escape(w), text, re.IGNORECASE):
            return True
    return False

# ---------------------------------------------------------------------------
# Known cities (all lower-case for matching)
# ---------------------------------------------------------------------------
_CITIES: list[str] = [
    "paris", "amsterdam", "tokyo", "bali", "delhi",
    "berlin", "london", "new york", "dubai", "singapore",
    "rome", "florence", "kyoto", "osaka", "ubud", "seminyak",
    "dehradun", "manali", "shimla",
]

# Canonical display names (title-cased)
_CITY_DISPLAY: dict[str, str] = {
    "ubud": "Bali",      # fixture is indexed by "bali"
    "seminyak": "Bali",
}


def _extract_city(message: str) -> str | None:
    """Return the first recognised city in *message*, or None."""
    msg = message.lower()
    for city in _CITIES:
        if city in msg:
            return _CITY_DISPLAY.get(city, city.title())
    return None


def _extract_amount(message: str) -> float | None:
    """Return the first dollar amount mentioned, e.g. '$1,850' → 1850.0."""
    m = re.search(r"\$\s*([\d,]+(?:\.\d{1,2})?)", message)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _extract_category(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["restaurant", "eat", "food", "ramen", "cuisine", "dinner", "lunch", "cafe"]):
        return "restaurants"
    if any(w in msg for w in ["attraction", "visit", "see", "sightseeing", "things to do",
                               "temple", "museum", "park", "observation", "deck"]):
        return "attractions"
    return "hotels"  # default


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------
class ToolCall(NamedTuple):
    name:   str
    args:   dict
    result: str   # JSON string returned by the tool


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
# Phrases that signal "I'm confirming a booking / asking to track spend"
# — no new search needed, skip all tool calls.
_BOOKING_RE = re.compile(
    r"let'?s assume|assume i book|assume that|let'?s say i"
    r"|please track|track that|add that|add both|add to"
    r"|book that|looks good|that works|looks solid"
    r"|i'?ve booked|i'?ve sorted|i'?ve confirmed"
    r"|booked at|confirm that|remind me",
    re.IGNORECASE,
)


def _is_booking_confirmation(message: str) -> bool:
    """True when the user is confirming a price / asking to track spend — not searching."""
    return bool(_BOOKING_RE.search(message))


def is_booking_confirmation(message: str) -> bool:
    """Public alias for GlobalState extraction and other callers."""
    return _is_booking_confirmation(message)


def extract_all_cities(message: str) -> list[str]:
    """
    Return recognised cities in *message* in order of first appearance,
    deduped by canonical display name (e.g. Ubud/Seminyak → Bali once).
    """
    msg = message.lower()
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for city in _CITIES:
        pos = msg.find(city)
        if pos < 0:
            continue
        disp = _CITY_DISPLAY.get(city, city.title())
        if disp in seen:
            continue
        seen.add(disp)
        hits.append((pos, disp))
    hits.sort(key=lambda h: h[0])
    return [h[1] for h in hits]


def route_and_call(user_message: str) -> list[ToolCall]:
    """
    Inspect *user_message* with keyword rules and immediately execute
    whatever tools are relevant.  Returns a list of ToolCall results.

    Booking-confirmation turns are skipped — they contain no search intent.
    Called BEFORE the LLM, so results can be injected as context.
    """
    if _is_booking_confirmation(user_message):
        return []

    msg = user_message.lower()
    calls: list[ToolCall] = []

    cities_in_msg = extract_all_cities(user_message)
    # For place-scoped tools (hotels / food / attractions / weather) we need a
    # single primary city.  For web_search queries we want every city mentioned.
    primary_city = cities_in_msg[0] if cities_in_msg else _extract_city(user_message)

    planning_signal = (
        _has_word(msg,
            "flight", "fly", "airline", "route", "ticket",
            "itinerary", "plan", "trip", "suggest", "recommend",
            "advice", "options", "budget",
        )
        or "multi-city" in msg
        or "travel from" in msg
        or "book flight" in msg
        or "what do you" in msg
        or "day by day" in msg
        or len(cities_in_msg) >= 2   # 2+ cities almost always means "plan the trip"
    )

    # ── web_search — flights / general travel info ────────────────────────
    if planning_signal:
        # Build a city-aware query so the fixture matcher actually hits a route
        # instead of silently falling back to "delhi to paris".
        if cities_in_msg:
            route_hint = " to ".join(c.lower() for c in cities_in_msg[:3])
            query = f"flights delhi to {route_hint}"[:140]
        else:
            query = f"flights {user_message[:120]}"
        result = dispatch_tool("web_search", {"query": query})
        calls.append(ToolCall("web_search", {"query": query}, result))

    # ── places_search — hotels / restaurants / attractions ────────────────
    city = primary_city
    if city:
        hotel_keywords = _has_word(msg,
            "hotel", "stay", "accommodation", "hostel", "resort",
            "villa", "inn", "lodge", "room",
        ) or "place to stay" in msg or "where to sleep" in msg

        # If the user is planning with cities+budget but didn't literally say
        # "hotel", we still want hotel fixtures for every mentioned city.
        implicit_hotel_ask = planning_signal and not _is_booking_confirmation(user_message)

        if hotel_keywords or implicit_hotel_ask:
            cat = "hotels"
            cities_to_query = cities_in_msg or [city]
            for c in cities_to_query[:3]:
                result = dispatch_tool("places_search", {"city": c, "category": cat})
                calls.append(ToolCall("places_search", {"city": c, "category": cat}, result))

        # Restaurants / food
        if _has_word(msg,
            "restaurant", "food", "ramen", "cuisine",
            "dinner", "lunch", "breakfast", "cafe", "dining",
        ) or "things to eat" in msg:
            cat = "restaurants"
            result = dispatch_tool("places_search", {"city": city, "category": cat})
            calls.append(ToolCall("places_search", {"city": city, "category": cat}, result))

        # Attractions
        if _has_word(msg,
            "attraction", "sightseeing", "temple", "museum", "landmark",
        ) or "things to do" in msg or "observation deck" in msg:
            cat = "attractions"
            result = dispatch_tool("places_search", {"city": city, "category": cat})
            calls.append(ToolCall("places_search", {"city": city, "category": cat}, result))

        # Weather
        if _has_word(msg,
            "weather", "temperature", "climate", "rain", "forecast",
        ) or "what to wear" in msg or "what to pack" in msg:
            result = dispatch_tool("weather_fetch", {"city": city})
            calls.append(ToolCall("weather_fetch", {"city": city}, result))

    return calls


# ---------------------------------------------------------------------------
# Context-block formatter
# ---------------------------------------------------------------------------
def format_context_block(calls: list[ToolCall]) -> str:
    """
    Render tool results as a human-readable context block to be injected
    into the conversation before the LLM reply.

    The block is prefixed and suffixed with clear markers so that —
    when we later build compression — it can be detected and summarised.
    """
    if not calls:
        return ""

    lines: list[str] = [
        "────────────────────────────────────────────────────",
        "TOOL RESULTS (fetched automatically — use these to answer):",
    ]
    for tc in calls:
        lines.append(f"\n[{tc.name}({json.dumps(tc.args, ensure_ascii=False)})]")
        # Pretty-print the JSON result
        try:
            parsed = json.loads(tc.result)
            lines.append(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            lines.append(tc.result)
    lines.append("\n────────────────────────────────────────────────────")

    return "\n".join(lines)

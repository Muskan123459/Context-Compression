"""
tools/places_search.py — Prompt tool: tells the model to generate real, accurate
place listings from its own training knowledge.
"""

SCHEMA = {
    "name": "places_search",
    "description": (
        "Find hotels, restaurants, or attractions in a location. "
        "Returns real listings with prices, ratings, and practical notes."
    ),
    "parameters": {
        "location":   "string — city or neighbourhood",
        "place_type": "string — hotel | restaurant | attraction",
        "query":      "string — optional keywords (e.g. 'budget', 'vegan', 'museum')",
    },
}

# Guidance templates per place type
_TEMPLATES = {
    "hotel": (
        "List 3 real hotels in {location}{query_note}. "
        "For each include: actual hotel name, realistic nightly price in USD (2024-25), "
        "neighbourhood, star rating, 2 amenities, and one honest practical note. "
        "Vary the price range from budget to mid-range unless 'luxury' was requested."
    ),
    "restaurant": (
        "List 3 real restaurants in {location}{query_note}. "
        "For each include: actual name, cuisine type, price range ($/$$/$$$/), "
        "neighbourhood, and one honest note about the food or atmosphere. "
        "If dietary restrictions were mentioned (vegan, shellfish allergy, etc.), "
        "flag compatibility clearly for each restaurant."
    ),
    "attraction": (
        "List 3 real attractions or activities in {location}{query_note}. "
        "For each include: actual name, entry fee in USD (0 if free), "
        "opening hours, and one essential tip (booking ahead, best time to visit, etc.)."
    ),
}


def run(location: str, place_type: str = "hotel", query: str = "") -> dict:
    pt       = place_type.lower().rstrip("s")   # "hotels" → "hotel"
    template = _TEMPLATES.get(pt, _TEMPLATES["attraction"])
    q_note   = f" (focus: {query})" if query.strip() else ""

    guidance = template.format(location=location.title(), query_note=q_note)

    return {
        "source":     "model_knowledge",
        "location":   location.title(),
        "place_type": pt,
        "guidance":   guidance,
    }

"""
tools/registry.py — Central dispatcher and prompt-block builder.

Importing this module is the only thing chat.py (or any agent) needs to do
to get access to all tools.

Public interface:
    SCHEMAS          — list of all tool schema dicts
    tool_schema_block() -> str   — formatted system-prompt section
    execute(name, arguments) -> str  — run a tool, return JSON string
"""

import json

from tools import budget_tracker, places_search, weather_fetch, web_search

# ---------------------------------------------------------------------------
# Schemas (one per tool, in declaration order)
# ---------------------------------------------------------------------------

SCHEMAS: list[dict] = [
    web_search.SCHEMA,
    places_search.SCHEMA,
    weather_fetch.SCHEMA,
    budget_tracker.SCHEMA,
]

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOLS: dict = {
    "web_search":     lambda args: web_search.run(**args),
    "places_search":  lambda args: places_search.run(**args),
    "weather_fetch":  lambda args: weather_fetch.run(**args),
    "budget_tracker": lambda args: budget_tracker.run(**args),
}


def execute(name: str, arguments: dict) -> str:
    """
    Call the named tool with the given arguments dict.

    For prompt tools (web_search, places_search) the result contains a
    'guidance' field — the agent loop should inject that as the TOOL_RESULT
    so the model fills in the answer from its own knowledge.

    For data tools (weather_fetch, budget_tracker) the result contains real
    data the model should use verbatim.

    Always returns a JSON string — never raises.
    """
    fn = _TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: '{name}'"})
    try:
        result = fn(arguments)
        return json.dumps(result, indent=2)
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for '{name}': {exc}"})
    except Exception as exc:
        return json.dumps({"error": f"Tool '{name}' raised: {exc}"})


# ---------------------------------------------------------------------------
# System-prompt block
# ---------------------------------------------------------------------------

def tool_schema_block() -> str:
    lines = [
        "## Your intelligence comes first",
        "",
        "Answer from your own knowledge whenever possible.",
        "Before considering a tool, ask yourself:",
        "  'Do I already know enough to give a good answer?'",
        "",
        "Answer DIRECTLY (no tool) for:",
        "  - General travel advice, tips, culture, etiquette",
        "  - Geography, history, language, cuisine overviews",
        "  - Packing lists, visa process explanations, safety tips",
        "  - Planning logic, itinerary structure, scheduling advice",
        "  - Budget math when numbers are already in the conversation",
        "  - Any factual question you can answer confidently",
        "",
        "Use a tool ONLY when you need data you genuinely cannot know:",
        "  - weather_fetch  → today's live weather in a specific city",
        "  - places_search  → actual hotel/restaurant/attraction listings",
        "  - web_search     → current flight prices, breaking travel news",
        "  - budget_tracker → recording or retrieving tracked spend numbers",
        "",
        "## Decision rule",
        "  CAN I answer this well from my own knowledge? → Answer directly.",
        "  Do I need a live number, listing, or tracked state?  → Use ONE tool.",
        "",
        "## Tool call format  (use EXACTLY this, nothing else on those lines)",
        "",
        "<tool_call>",
        '{"name": "<tool_name>", "arguments": {<json key-value pairs>}}',
        "</tool_call>",
        "",
        "## After receiving a TOOL_RESULT",
        "  - Synthesise it — do NOT just echo the raw data.",
        "  - Layer in your own expertise: tips, warnings, context, recommendations.",
        "  - Surface any warnings (over budget, shellfish, scheduling conflicts) first.",
        "  - Always end with a clear recommendation or next step.",
        "",
        "## Tool definitions",
    ]
    for schema in SCHEMAS:
        lines.append(f"\n### {schema['name']}")
        lines.append(f"Description: {schema['description']}")
        lines.append("Parameters:")
        for param, desc in schema["parameters"].items():
            lines.append(f"  - {param}: {desc}")
    return "\n".join(lines)

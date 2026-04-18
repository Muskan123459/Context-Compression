"""
Tool registry — OpenAI-style schema list + a single dispatcher that routes
tool calls by name to the individual tool modules.
"""
from __future__ import annotations

from ._common import _json
from .budget_tracker import SCHEMA as BUDGET_TRACKER_SCHEMA
from .budget_tracker import budget_tracker
from .places_search import SCHEMA as PLACES_SEARCH_SCHEMA
from .places_search import places_search
from .read_memory import SCHEMA as READ_MEMORY_SCHEMA
from .read_memory import read_memory
from .weather_fetch import SCHEMA as WEATHER_FETCH_SCHEMA
from .weather_fetch import weather_fetch
from .web_search import SCHEMA as WEB_SEARCH_SCHEMA
from .web_search import web_search


TOOL_SCHEMAS: list[dict] = [
    WEB_SEARCH_SCHEMA,
    PLACES_SEARCH_SCHEMA,
    WEATHER_FETCH_SCHEMA,
    BUDGET_TRACKER_SCHEMA,
    READ_MEMORY_SCHEMA,
]


def dispatch_tool(name: str, args: dict) -> str:
    """Route a tool call to the appropriate function and return its JSON string."""
    try:
        if name == "web_search":
            result = web_search(**args)
        elif name == "places_search":
            result = places_search(**args)
        elif name == "weather_fetch":
            result = weather_fetch(**args)
        elif name == "budget_tracker":
            result = budget_tracker(**args)
        elif name == "read_memory":
            result = read_memory(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc), "tool": name, "args": args}
    return _json(result)

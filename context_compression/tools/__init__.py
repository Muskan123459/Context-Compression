"""
v1 tools package — four mock travel-agent tools with 2 000-3 000 token
JSON fixtures. Each tool is deterministic / rule-based; no real APIs.

Public surface (backward-compatible with the old flat `tools.py`):

    from tools import (
        dispatch_tool,         # routes a tool call by name
        TOOL_SCHEMAS,          # OpenAI-style function schemas
        web_search,            # Tool 1 — flights and general travel info
        places_search,         # Tool 2 — hotels / restaurants / attractions
        weather_fetch,         # Tool 3 — detailed weather + packing guide
        budget_tracker,        # Tool 4 — stateful spend ledger
    )

Each tool also lives in its own module (e.g. `tools.web_search`) so the
fixtures can be edited in isolation.
"""
from __future__ import annotations

from .budget_tracker import budget_tracker
from .places_search import places_search
from .read_memory import read_memory
from .registry import TOOL_SCHEMAS, dispatch_tool
from .weather_fetch import weather_fetch
from .web_search import web_search

__all__ = [
    "TOOL_SCHEMAS",
    "budget_tracker",
    "dispatch_tool",
    "places_search",
    "read_memory",
    "weather_fetch",
    "web_search",
]

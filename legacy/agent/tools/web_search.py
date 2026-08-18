"""
tools/web_search.py — Prompt tool: injects a directive for the model to answer
from its own knowledge, supplemented by live DDG data when available.
"""

import json
import urllib.parse
import urllib.request

SCHEMA = {
    "name": "web_search",
    "description": (
        "Search for current flight prices, travel advisories, visa requirements, "
        "transportation options, or any factual travel query."
    ),
    "parameters": {
        "query": "string — the search query",
    },
}

_DDG_URL = "https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"


def _ddg(query: str) -> str | None:
    """Return a short live snippet from DDG, or None."""
    try:
        q   = urllib.parse.quote(query)
        req = urllib.request.Request(
            _DDG_URL.format(q=q),
            headers={"User-Agent": "travel-agent/1.0"},
        )
        raw = urllib.request.urlopen(req, timeout=5).read()
        d   = json.loads(raw)
        if d.get("AbstractText"):
            return d["AbstractText"][:400]
        if d.get("Answer"):
            return d["Answer"]
    except Exception:
        pass
    return None


def run(query: str) -> dict:
    live = _ddg(query)
    if live:
        return {
            "source":   "duckduckgo",
            "snippet":  live,
            "guidance": (
                f"You found this live snippet for '{query}': {live}\n"
                "Use it alongside your own knowledge to give a complete, accurate answer."
            ),
        }
    # No live data — return a prompt directive
    return {
        "source":   "model_knowledge",
        "guidance": (
            f"No live data available for: '{query}'.\n"
            "Answer from your own knowledge. Be specific: include real prices, "
            "timings, or facts where you know them. If uncertain, say so clearly."
        ),
    }

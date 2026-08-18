"""Shared helpers used by every tool module."""
from __future__ import annotations

import json


def _json(d: dict) -> str:
    """Serialise a dict to a JSON string (the format the agent injects as tool output)."""
    return json.dumps(d, ensure_ascii=False, indent=2)

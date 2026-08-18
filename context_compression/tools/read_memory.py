"""
TOOL 5 — read_memory

Counterpart to the L2 offload path.  When `route_and_call` decides a tool result
is too large to keep inline, it writes the raw JSON to `memory_store/<name>.json`
and injects a short preview + path marker into the chat.  The LLM can then call
`read_memory(path="…")` to pull the full payload back in on demand.

Safety: the path argument MUST resolve under `MEMORY_STORE_DIR`.  Anything else
(absolute paths, `..` traversal, symlinks escaping the store) is rejected.
"""
from __future__ import annotations

from pathlib import Path

from ..config import CONVERSATION_STORE_DIR, MEMORY_STORE_DIR


def _resolve_safe(path_arg: str) -> Path:
    """Resolve *path_arg* and assert it lives under one of the allowed stores.

    Accepts any of the path shapes we might see in the wild:
      - absolute: `/abs/.../memory_store/foo.json`
      - relative to the v1 dir: `memory_store/foo.json`  (what offload advertises)
                                `conversation_history/thread.md` (what compaction advertises)
      - bare filename: `foo.json`  — assumed to live in MEMORY_STORE_DIR
    """
    MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSATION_STORE_DIR.mkdir(parents=True, exist_ok=True)

    mem = MEMORY_STORE_DIR.resolve()
    conv = CONVERSATION_STORE_DIR.resolve()
    allowed_roots = (mem, conv)

    s = (path_arg or "").strip()
    if not s:
        raise ValueError("empty path")

    raw = Path(s)
    if raw.is_absolute():
        candidate = raw
    else:
        parts = list(raw.parts)
        # Route by the advertised prefix, then strip it.
        if parts and parts[0] == conv.name:
            candidate = conv.joinpath(*parts[1:]) if parts[1:] else conv
        else:
            stripped = parts
            while stripped and stripped[0] == mem.name:
                stripped = stripped[1:]
            candidate = mem.joinpath(*stripped) if stripped else mem

    try:
        resolved = candidate.resolve(strict=False)
    except Exception as exc:
        raise ValueError(f"could not resolve path: {exc}") from exc

    for root in allowed_roots:
        if resolved == root or root in resolved.parents:
            return resolved

    raise ValueError(
        f"path '{path_arg}' escapes allowed stores "
        f"({mem}, {conv}) — refusing to read."
    )


def read_memory(path: str) -> dict:
    """Return the raw contents of an offloaded tool dump.

    Parameters
    ----------
    path : str
        Either an absolute path inside MEMORY_STORE_DIR, or a filename relative
        to MEMORY_STORE_DIR (what the offload pointer advertises).

    Returns
    -------
    dict — one of:
        {"path": str, "bytes": int, "content": str}   on success
        {"error": str, "path": str}                   on failure
    """
    try:
        resolved = _resolve_safe(path)
    except ValueError as exc:
        return {"error": str(exc), "path": path}

    if not resolved.exists():
        return {"error": "file not found", "path": str(resolved)}

    try:
        text = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(resolved),
        }

    return {
        "path": str(resolved),
        "bytes": len(text.encode("utf-8")),
        "content": text,
    }


SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": (
            "Read the full contents of an offloaded tool dump from the agent's "
            "memory store. Use this ONLY when a previous tool result in the "
            "conversation was truncated with a <TRUNCATED> marker and you need "
            "details that weren't in the preview."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The exact path advertised by the <TRUNCATED> marker "
                        "(e.g. 'memory_store/places_search_ab12cd.json')."
                    ),
                }
            },
            "required": ["path"],
        },
    },
}

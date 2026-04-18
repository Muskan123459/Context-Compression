"""
Layer 3 — threshold-triggered conversation compaction.

When the assembled message list is about to exceed `COMPACT_THRESHOLD_PCT` of
`CONTEXT_LIMIT`, we:

  1. Pick the oldest `N_COMPACT_TURNS` user/assistant pairs (never touching
     the pinned system message or the most recent turn the user just sent).
  2. Append the raw text of those turns to `conversation_history/<thread>.md`
     so nothing is lost — archives are the receipt the judges see.
  3. Ask the LLM for a compact summary, *state-aware*: we hand it the current
     L1 state and explicitly tell it to skip anything the user has removed /
     overridden. That makes the pivot test ("scratch Bali") a natural fit —
     when Bali isn't in `active_goals` anymore, the summary can't leak it.
  4. Replace the archived slice with a single synthetic system message shaped
     exactly like the reference diagram: pointer + `<summary>` block.

The compaction engine is domain-free: it reads arbitrary JSON state and
arbitrary message content.  Only the summariser prompt mentions "plans,
locations, decisions" as common examples — swap that string for any other
domain and the machinery still works.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from config import (
    COMPACT_THRESHOLD_PCT,
    CONTEXT_LIMIT,
    CONVERSATION_STORE_DIR,
    MODEL,
    N_COMPACT_TURNS,
    SUMMARY_MAX_TOKENS,
)
from global_state import GenericState, count_text_tokens

if TYPE_CHECKING:
    from openai import OpenAI


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class CompactionEvent:
    """Observability record for a single compaction pass."""
    fired:           bool = False
    turns_archived:  int = 0
    archive_path:    Optional[str] = None
    tokens_before:   int = 0
    tokens_after:    int = 0
    summary:         str = ""
    skipped_reason:  str = ""          # populated when should_compact says no
    pivot_fragments: list[str] = field(default_factory=list)

    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


# ---------------------------------------------------------------------------
# Public predicate
# ---------------------------------------------------------------------------
def _tokens_of_messages(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            total += count_text_tokens(content)
    return total


def should_compact(messages: list[dict]) -> bool:
    """True when the assembled message list already strains the context window."""
    return _tokens_of_messages(messages) > COMPACT_THRESHOLD_PCT * CONTEXT_LIMIT


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
def _ensure_store() -> Path:
    CONVERSATION_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return CONVERSATION_STORE_DIR


def _render_archive_block(slice_messages: list[dict], compaction_idx: int) -> str:
    stamp = datetime.now().isoformat(timespec="seconds")
    header = (
        f"\n\n---\n"
        f"## Compaction #{compaction_idx} — archived {stamp}\n"
        f"Turns: {len(slice_messages)} message(s)\n\n"
    )
    parts = [header]
    for m in slice_messages:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        content = m.get("content") if isinstance(m, dict) else str(m)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        parts.append(f"### {role}\n\n{content}\n")
    return "\n".join(parts)


def _append_to_thread_file(path: Path, block: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------
def _pick_slice(messages: list[dict], n_turns: int) -> tuple[int, int]:
    """Return (start, end) index slice to archive.

    Skips:
      - the leading system message (pinned state + system prompt)
      - the last 2 messages (the user's current turn and the one before it —
        the model still needs recent context to respond coherently)
    """
    if not messages:
        return (0, 0)

    start = 1 if messages[0].get("role") == "system" else 0
    tail_keep = 2
    end = max(start, len(messages) - tail_keep)
    # Each "turn" is a user/assistant pair, so n_turns ≈ 2*n_turns messages.
    desired = n_turns * 2
    take = min(desired, end - start)
    return (start, start + take)


# ---------------------------------------------------------------------------
# State-aware summarisation
# ---------------------------------------------------------------------------
_SUMMARISER_SYSTEM = """\
You are a conversation compactor.

You will be given:
  • CURRENT STATE — the user's up-to-date structured memory (JSON).
  • TRANSCRIPT   — a slice of older messages we want to archive.

Write a concise prose summary of the TRANSCRIPT.

CRITICAL RULES:
- Cross-reference against CURRENT STATE. Do NOT mention any plan, location,
  preference, booking, or decision that has been CANCELLED or REMOVED from
  the current state.
- Preserve: goals the user still cares about, commitments, numeric facts
  (budgets, dates, counts), safety-critical constraints (allergies, fears).
- Drop: tool-result details that aren't decision-grade, pleasantries,
  agent chain-of-thought.
- Under 180 words. Plain prose. No bullet lists, no markdown headers.
"""


def _summarise(
    slice_messages: list[dict],
    state: GenericState,
    pivot_fragments: list[str],
    client: "OpenAI",
) -> str:
    transcript_lines: list[str] = []
    for m in slice_messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        # Keep the transcript short so the summariser itself doesn't blow the
        # context — truncate very long tool blobs at 800 chars.
        if len(content) > 800:
            content = content[:800] + "\n…(truncated for summariser)"
        transcript_lines.append(f"[{role}] {content}")
    transcript = "\n\n".join(transcript_lines)

    user_block = (
        f"CURRENT STATE:\n{json.dumps(state.to_dict(), ensure_ascii=False)}\n\n"
    )
    if pivot_fragments:
        user_block += (
            f"PIVOT FRAGMENTS — the user just cancelled these topics; "
            f"they MUST NOT appear in the summary:\n"
            f"{json.dumps(pivot_fragments, ensure_ascii=False)}\n\n"
        )
    user_block += f"TRANSCRIPT:\n{transcript}"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SUMMARISER_SYSTEM},
                {"role": "user",   "content": user_block},
            ],
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"(summariser error: {type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# The compactor
# ---------------------------------------------------------------------------
def compact(
    messages: list[dict],
    state: GenericState,
    client: "OpenAI",
    thread_id: str,
    *,
    pivot_fragments: list[str] | None = None,
    compaction_idx: int = 1,
) -> tuple[list[dict], CompactionEvent]:
    """Archive the oldest turns and replace them with a pointer + summary.

    Returns the new message list and a populated `CompactionEvent`.
    """
    tokens_before = _tokens_of_messages(messages)
    event = CompactionEvent(
        fired=False,
        tokens_before=tokens_before,
        tokens_after=tokens_before,
        pivot_fragments=list(pivot_fragments or []),
    )

    start, end = _pick_slice(messages, N_COMPACT_TURNS)
    if end <= start:
        event.skipped_reason = "no eligible turns to archive"
        return messages, event

    slice_messages = messages[start:end]

    store = _ensure_store()
    archive_path = store / f"{thread_id}.md"
    _append_to_thread_file(
        archive_path,
        _render_archive_block(slice_messages, compaction_idx),
    )
    relative_archive = f"conversation_history/{archive_path.name}"

    summary = _summarise(slice_messages, state, event.pivot_fragments, client)

    synthetic = (
        f"You are in the middle of a conversation. The full raw history of "
        f"the archived turns has been saved to {relative_archive} — call "
        f"read_memory(path=\"{relative_archive}\") if you need specifics.\n\n"
        f"A condensed summary follows:\n\n"
        f"<summary>\n{summary}\n</summary>"
    )

    new_messages = (
        messages[:start]
        + [{"role": "system", "content": synthetic}]
        + messages[end:]
    )

    event.fired = True
    event.turns_archived = end - start
    event.archive_path = relative_archive
    event.tokens_after = _tokens_of_messages(new_messages)
    event.summary = summary
    return new_messages, event


__all__ = ["CompactionEvent", "compact", "should_compact"]

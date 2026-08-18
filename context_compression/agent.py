"""
Core agent loop — wires together the three compression layers.

Every turn does, in order:

  1. L1  — state_extractor.run_extractor(...) updates the hybrid-schema
           GenericState (regex safety-net → LLM ops → pivot purge).
  2.     — rule-based tool router pre-calls tools; results > threshold are
           written to `memory_store/` and replaced inline with a <TRUNCATED>
           marker + `read_memory(...)` pointer (L2 offload).
  3.     — deterministic budget-facts block is injected ahead of tool results
           whenever `state.financial_constraints` has numeric fields.
  4. L3  — prior user+assistant turns are compressed every turn according to
           HISTORY_MODE (default "summary" = ONE LLM call folds them into a
           single <history_summary> block; also "distill" = per-reply
           1-liners, or "none" = raw passthrough). Additionally, if the
           assembled message list exceeds COMPACT_THRESHOLD_PCT of
           CONTEXT_LIMIT, the oldest turns are archived to disk and replaced
           with a pointer + state-aware `<summary>` block.
  5.     — the assembled prompt is sent to SmolLM3 with the full tool schema
           list, so the model may emit `read_memory(...)` to pull any
           offloaded payload back in. One tool-call round is allowed per turn.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .compaction import CompactionEvent, compact, should_compact
from .config import (
    CONTEXT_LIMIT,
    ENABLE_THINKING,
    HISTORY_MODE,
    HISTORY_SUMMARY_MAX_TOKENS,
    HISTORY_SUMMARY_MIN_MSGS,
    MAX_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    VLLM_BASE_URL,
    WARN_THRESHOLD,
)
from .global_state import GenericState, build_budget_facts_block, count_text_tokens
from .state_extractor import run_extractor
from .tool_router import ToolCall, format_context_block, route_and_call
from .tools import TOOL_SCHEMAS, dispatch_tool


# ---------------------------------------------------------------------------
# OpenAI-compatible client (talks to vLLM)
# ---------------------------------------------------------------------------
client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class TokenUsage:
    """Cumulative token stats returned alongside every reply."""
    prompt: int = 0
    completion: int = 0
    total: int = 0
    context_pct: float = 0.0  # total / CONTEXT_LIMIT * 100


@dataclass
class AgentReply:
    text:                 str
    usage:                TokenUsage
    trace:                list[str]       = field(default_factory=list)
    tool_calls:           list[ToolCall]  = field(default_factory=list)
    augmented_message:    str             = ""
    state:                GenericState    = field(default_factory=GenericState)
    pinned_block_tokens:  int             = 0
    compaction:           Optional[CompactionEvent] = None
    read_memory_calls:    list[dict]      = field(default_factory=list)
    # One snapshot per vLLM request this turn (main call, then read_memory follow-ups).
    llm_prompt_rounds:    list[list[dict[str, str]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Naked tool-call JSON stripper
#
# Small / under-tuned models occasionally emit Hermes-style function calls as
# *plain text* inside `message.content` instead of the proper `tool_calls`
# field. When that happens vLLM's hermes parser doesn't intercept them (the
# `<tool_call>` tags are missing) and they render verbatim in the UI, e.g.
#
#   {"name": "budget_tracker", "arguments": {"action": "add_expense", ...}}
#   {"name": "budget_tracker", "arguments": {"action": "add_expense", ...}}
#   Balance remaining: USD 2,530
#
# This is never a useful output — the tool was either already executed by the
# rule-based router or isn't callable from the model path at all. Strip these
# blobs (and any `<tool_call>…</tool_call>` tags that slip through) before
# handing the reply back to the UI.
# ---------------------------------------------------------------------------
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
# Matches a top-level JSON object that has a "name" key and either "arguments"
# or "parameters" — covers Qwen/Hermes/OpenAI-style leaks. Non-greedy on the
# outer braces; requires at least one nested "{" to avoid eating ordinary prose.
_NAKED_TOOL_JSON_RE = re.compile(
    r"""\{\s*                                       # opening brace
        "(?:name|function)"\s*:\s*"[^"]+"\s*,\s*    # "name": "..."
        "(?:arguments|parameters)"\s*:\s*\{.*?\}    # "arguments": {...}
        \s*\}""",
    re.DOTALL | re.VERBOSE,
)


def _strip_tool_call_leak(text: str) -> tuple[str, int]:
    """Remove stray tool-call JSON from a model reply.

    Returns (cleaned_text, num_blobs_stripped). Safe to call on every reply —
    if nothing matches, the text is returned unchanged.
    """
    if not text:
        return text, 0
    count = 0
    cleaned, n = _TOOL_CALL_TAG_RE.subn("", text)
    count += n
    cleaned, n = _NAKED_TOOL_JSON_RE.subn("", cleaned)
    count += n
    if count:
        # Collapse the blank lines the substitutions leave behind.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, count


# ---------------------------------------------------------------------------
# <think> stripping (SmolLM3 occasionally leaks reasoning traces)
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> tuple[str, str]:
    think_parts: list[str] = []

    def _capture(match: re.Match) -> str:
        think_parts.append(match.group(1).strip())
        return ""

    clean = re.sub(r"<think>(.*?)</think>", _capture, text, flags=re.DOTALL)

    open_idx = clean.find("<think>")
    if open_idx != -1:
        think_parts.append(clean[open_idx + len("<think>"):].strip())
        clean = clean[:open_idx]

    clean = clean.strip()
    if not clean and think_parts:
        clean = (
            "_(model ran out of tokens while thinking — raise `MAX_TOKENS` "
            "or keep `/no_think` in the system prompt)_"
        )

    return clean, "\n".join(p for p in think_parts if p)


def _token_bar(usage: TokenUsage) -> str:
    filled = int(usage.context_pct / 5)  # 20-char bar at 100 %
    bar = "█" * filled + "░" * (20 - filled)
    warn = " ⚠ NEAR LIMIT" if usage.context_pct >= WARN_THRESHOLD * 100 else ""
    return (
        f"[tokens] prompt={usage.prompt:,}  completion={usage.completion:,}  "
        f"total={usage.total:,}  [{bar}] {usage.context_pct:.1f}% of {CONTEXT_LIMIT:,}{warn}"
    )


# ---------------------------------------------------------------------------
# read_memory fallback parser (SmolLM3 sometimes emits plain-text calls)
# ---------------------------------------------------------------------------
_PLAINTEXT_RM_RE = re.compile(
    r"read_memory\s*\(\s*(?:path\s*=\s*)?[\"']([^\"']+)[\"']\s*\)",
    re.IGNORECASE,
)


def _extract_plaintext_read_memory(text: str) -> Optional[str]:
    m = _PLAINTEXT_RM_RE.search(text or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# History distillation — stop the self-reinforcing parroting loop.
#
# Problem: Gradio appends each full assistant reply to `history`, and the next
# turn's prompt sees that reply as an in-context example of what to output.
# On a 3B model the in-context example dominates the system-prompt rules and
# the model just re-emits the same canned block (see turn-003.txt evidence).
#
# Fix: before building the model prompt, replace every prior assistant reply
# with a 1-line neutral abstract. The Gradio UI still shows the full reply to
# the user; only the model-visible history is compressed. Cached by content
# hash so we spend ≤1 distillation call per *new* assistant reply (not N per
# turn). Toggle with DISTILL_HISTORY=0 to disable for debugging.
# ---------------------------------------------------------------------------
_DISTILL_CACHE: dict[str, str] = {}

_DISTILL_SYSTEM = """/no_think

You compress ONE assistant reply into a single neutral sentence for use as
conversation context.

Output rules:
- Exactly ONE line. No bullets, no headers, no markdown, no blank lines.
- At most 25 words.
- Describe what the assistant DID this turn (offered options, asked a
  question, confirmed a booking) — NOT the formatting or the full fact dump.
- Never use the word "Confirmed:" unless the assistant actually recorded a
  booking in that turn.
- Do not restate numbers, hotel names, or itinerary details verbatim.
- Plain text only. No <think>, no JSON, no quotes around the sentence."""


def _coerce_to_text(content: Any) -> str:
    """Flatten Gradio's list-of-parts content shape to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text") or c.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return ""


def _snapshot_messages(messages: list[dict]) -> list[dict[str, str]]:
    """Copy messages as plain role/content strings for UI / debugging."""
    snap: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "?")
        text = _coerce_to_text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            try:
                tc_blob = json.dumps(m["tool_calls"], ensure_ascii=False, indent=2)
            except Exception:
                tc_blob = repr(m.get("tool_calls"))
            text = (text + "\n\n[tool_calls]\n" if text else "[tool_calls]\n") + tc_blob
        elif role == "tool":
            name = m.get("name") or ""
            tid = m.get("tool_call_id") or ""
            text = f"(tool result · {name} · id={tid})\n{text}"
        snap.append({"role": role, "content": text})
    return snap


def _distill_one_reply(text: str, client: OpenAI) -> str:
    """1-line abstract of a prior assistant reply. Cached by content hash."""
    if not text or len(text) <= 120:
        return text
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cached = _DISTILL_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _DISTILL_SYSTEM},
                {"role": "user",   "content": text[:2000]},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw = ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    first_line = raw.split("\n", 1)[0].strip() if raw else ""
    short = first_line[:300] or f"[prior assistant reply, {len(text)} chars — distillation failed]"
    _DISTILL_CACHE[key] = short
    return short


def _distill_history(
    history: list[dict],
    client: OpenAI,
    trace: list[str],
) -> list[dict]:
    """Return a copy of history where each assistant turn is a 1-line abstract.

    User / tool messages pass through untouched. Short replies (≤120 chars) are
    left alone — they're already fine.
    """
    if os.getenv("DISTILL_HISTORY", "1") == "0":
        return list(history)

    out: list[dict] = []
    distilled = 0
    cache_hits = 0
    for msg in history:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        raw_text = _coerce_to_text(msg.get("content"))
        if not raw_text or len(raw_text) <= 120:
            out.append({"role": "assistant", "content": raw_text})
            continue
        key = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
        had_cache = key in _DISTILL_CACHE
        short = _distill_one_reply(raw_text, client)
        out.append({"role": "assistant", "content": short})
        distilled += 1
        if had_cache:
            cache_hits += 1

    if distilled:
        trace.append(
            f"history distilled: {distilled} assistant turn(s) compressed "
            f"(cache hits: {cache_hits}/{distilled})"
        )
    return out


# ---------------------------------------------------------------------------
# Per-turn history summarisation (Layer-3 "always on" mode)
#
# Distillation-only (above) compresses each assistant reply in isolation but
# leaves every prior user message verbatim. That's still O(N) lines of past
# turns at turn N. This summariser takes the whole prior user+assistant
# transcript and folds it into ONE `<history_summary>` block via a single
# LLM call per turn (cached by content hash, so repeated turns cost zero).
#
# The model's view of the conversation on turn N becomes:
#   [system]            SYSTEM_PROMPT + pinned GenericState block
#   [system-summary]    <history_summary> covering turns 1 .. N-1
#   [user]              turn-N message (+ budget facts + tool results)
#
# Falls back to raw passthrough when there are fewer than
# HISTORY_SUMMARY_MIN_MSGS prior messages (not worth an LLM call on turn 1).
# ---------------------------------------------------------------------------
_HISTORY_SUMMARY_CACHE: dict[str, str] = {}

_HISTORY_SUMMARY_SYSTEM = """/no_think

You compress a multi-turn conversation transcript into ONE concise summary
that a downstream agent will read as its only record of prior turns.

Output rules:
- Plain prose, 4-10 sentences, under 220 words total.
- Preserve: what the user asked for, facts they gave (names, numbers,
  dates, locations, preferences, hard constraints, allergies), what the
  assistant offered or proposed, any commitments or bookings made, any
  corrections or pivots the user issued.
- Drop: pleasantries, filler, agent chain-of-thought, verbatim tool dumps,
  exact option lists (summarise as "offered 3 Tokyo hotel options" etc.).
- If the user cancelled or superseded something earlier, say so explicitly
  ("user initially asked for Bali, then switched to Switzerland").
- Do not invent facts not present in the transcript.
- No markdown headers, no bullets, no code fences, no <think> blocks,
  no quotes around the summary."""


def _flatten_history_for_summary(history: list[dict]) -> str:
    """Render user+assistant turns as a plain transcript for the summariser."""
    lines: list[str] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _coerce_to_text(msg.get("content"))
        if not text:
            continue
        if len(text) > 1500:
            text = text[:1500] + " …(truncated for summariser)"
        lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


def _summarize_history(
    history: list[dict],
    client: OpenAI,
    trace: list[str],
) -> list[dict]:
    """Fold ALL prior user+assistant turns into ONE synthetic summary message.

    One LLM call per turn (cached by transcript hash). Messages with a role
    other than user/assistant (e.g. tool results) are passed through unchanged
    and appear after the summary block so the model can still reference them.
    """
    summarisable = [
        m for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    passthrough = [
        m for m in history
        if isinstance(m, dict) and m.get("role") not in ("user", "assistant")
    ]

    if len(summarisable) < HISTORY_SUMMARY_MIN_MSGS:
        return list(history)

    transcript = _flatten_history_for_summary(summarisable)
    if not transcript:
        return list(history)

    key = hashlib.sha1(transcript.encode("utf-8")).hexdigest()
    cached = _HISTORY_SUMMARY_CACHE.get(key)
    had_cache = cached is not None

    if cached is None:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _HISTORY_SUMMARY_SYSTEM},
                    {"role": "user",   "content": transcript[:12000]},
                ],
                max_tokens=HISTORY_SUMMARY_MAX_TOKENS,
                temperature=0.2,
            )
            cached = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            cached = f"(history summariser error: {type(exc).__name__}: {exc})"
        cached = re.sub(r"<think>.*?</think>", "", cached, flags=re.DOTALL).strip()
        if not cached:
            cached = (
                f"[prior conversation: {len(summarisable)} turn(s) — "
                f"summariser returned empty]"
            )
        _HISTORY_SUMMARY_CACHE[key] = cached

    summary_tokens = count_text_tokens(cached)
    trace.append(
        f"history summarised: {len(summarisable)} turn(s) → 1 "
        f"<history_summary> block ({summary_tokens} tok, cache hit: {had_cache})"
    )

    synthetic = {
        "role": "system",
        "content": (
            "Prior conversation summary — this is your only record of the "
            "user+assistant turns before the current one. Treat it as "
            "authoritative, but prefer the pinned Current state block above "
            "when they conflict.\n\n"
            f"<history_summary>\n{cached}\n</history_summary>"
        ),
    }
    return [synthetic, *passthrough]


def _prepare_history_for_model(
    history: list[dict],
    client: OpenAI,
    trace: list[str],
) -> list[dict]:
    """Dispatch to the configured history-compression mode (HISTORY_MODE)."""
    # Legacy escape hatch still supported: DISTILL_HISTORY=0 forces raw passthrough.
    if os.getenv("DISTILL_HISTORY", "1") == "0":
        trace.append("history mode: none (DISTILL_HISTORY=0 override)")
        return list(history)

    mode = (os.getenv("HISTORY_MODE") or HISTORY_MODE).lower()
    if mode == "none":
        trace.append("history mode: none (raw passthrough)")
        return list(history)
    if mode == "distill":
        trace.append("history mode: distill (per-reply 1-liners)")
        return _distill_history(history, client, trace)
    # Default and explicit "summary".
    trace.append("history mode: summary (single LLM summary of all prior turns)")
    return _summarize_history(history, client, trace)


# ---------------------------------------------------------------------------
# Prompt dump for debugging
# ---------------------------------------------------------------------------
def _dump_prompt(messages: list[dict], turn_no: int, trace: list[str]) -> None:
    if os.getenv("DUMP_PROMPT", "1") == "0":
        return
    try:
        logs_dir = Path(__file__).with_name("logs")
        logs_dir.mkdir(exist_ok=True)
        dump_path = logs_dir / f"turn-{turn_no:03d}.txt"
        with dump_path.open("w", encoding="utf-8") as fh:
            fh.write(f"=== Turn {turn_no} — full prompt sent to vLLM ===\n")
            fh.write(
                f"messages: {len(messages)} items "
                f"(system + history + current user)\n\n"
            )
            for i, m in enumerate(messages):
                role = m.get("role", "?") if isinstance(m, dict) else "?"
                content = m.get("content") if isinstance(m, dict) else m
                if isinstance(content, str):
                    body = content
                elif content is None:
                    body = "<none>"
                else:
                    try:
                        body = json.dumps(content, ensure_ascii=False, indent=2)
                    except Exception:
                        body = repr(content)
                fh.write(f"── [{i}] role={role} ──\n")
                fh.write(body)
                fh.write("\n\n")
        trace.append(
            f"full prompt dumped → logs/turn-{turn_no:03d}.txt "
            f"({len(messages)} messages)"
        )
    except Exception as exc:
        trace.append(f"prompt dump skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tool-call handling on the response path (read_memory)
# ---------------------------------------------------------------------------
def _handle_tool_calls(
    response_msg: Any,
    trace: list[str],
    read_memory_calls: list[dict],
) -> list[dict]:
    """Turn an OpenAI tool_calls response into follow-up messages.

    Only `read_memory` is honoured; anything else is converted into a
    structured error so the model doesn't silently hang.
    """
    tool_calls = getattr(response_msg, "tool_calls", None) or []
    if not tool_calls:
        return []

    follow_up: list[dict] = [
        {
            "role": "assistant",
            "content": response_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
    ]

    for tc in tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        if name != "read_memory":
            tool_result = json.dumps(
                {"error": f"tool '{name}' is not callable from the LLM in this loop"},
                ensure_ascii=False,
            )
        else:
            tool_result = dispatch_tool("read_memory", args)
            read_memory_calls.append({"args": args, "bytes": len(tool_result)})
            trace.append(
                f"read_memory ← {args.get('path', '?')} "
                f"({len(tool_result):,} chars returned)"
            )

        follow_up.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_result,
            }
        )

    return follow_up


def _handle_plaintext_read_memory(
    raw_text: str,
    trace: list[str],
    read_memory_calls: list[dict],
) -> Optional[dict]:
    """Fallback when SmolLM3 emits `read_memory("...")` as plain text."""
    path = _extract_plaintext_read_memory(raw_text)
    if not path:
        return None
    tool_result = dispatch_tool("read_memory", {"path": path})
    read_memory_calls.append({"args": {"path": path}, "bytes": len(tool_result)})
    trace.append(
        f"read_memory (plain-text fallback) ← {path} "
        f"({len(tool_result):,} chars returned)"
    )
    return {
        "role": "user",
        "content": (
            f"(system — read_memory auto-fulfilled from plain-text call)\n"
            f"{tool_result}"
        ),
    }


# ---------------------------------------------------------------------------
# Thinking-mode toggle helpers
#
# When ENABLE_THINKING=1:
#   • We strip the `/no_think` directive and the "no chain-of-thought" clause
#     from the system prompt so we stop actively suppressing reasoning.
#   • We pass `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`
#     to the main LLM call — this is the native Qwen3 / QwQ toggle that tells
#     the chat template to emit a `<think>` block before the answer. vLLM
#     silently drops unknown kwargs, so on Qwen2.5-Instruct the call still
#     succeeds; the model just won't actually think unless the prompt coaxes
#     it (`_strip_think` is already set up to handle both cases).
# ---------------------------------------------------------------------------
_NO_THINK_DIRECTIVE_RE = re.compile(r"^\s*/no_think\s*\n*", re.IGNORECASE)
_NO_COT_LINE_RE = re.compile(
    r"^\s*-\s*Reply directly as plain prose.*?no chain-of-thought\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _system_prompt_for_model() -> str:
    """SYSTEM_PROMPT, with anti-thinking hints removed when thinking is on."""
    if not ENABLE_THINKING:
        return SYSTEM_PROMPT
    prompt = _NO_THINK_DIRECTIVE_RE.sub("", SYSTEM_PROMPT, count=1)
    prompt = _NO_COT_LINE_RE.sub("", prompt)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    thinking_rule = (
        "- You may reason step-by-step inside a single `<think>...</think>` "
        "block before your reply; keep the final answer after the closing "
        "tag brief and user-facing."
    )
    return f"{prompt}\n{thinking_rule}"


def _thinking_extra_body() -> dict[str, Any]:
    """extra_body kwargs passed only to user-visible LLM calls."""
    if not ENABLE_THINKING:
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


# ---------------------------------------------------------------------------
# Core agentic loop
# ---------------------------------------------------------------------------
def run_agent(
    user_message: str,
    history: list[dict],
    *,
    state: GenericState | None = None,
    verbose: bool = True,
) -> AgentReply:
    """
    Full 3-layer turn:
      L1 extractor → pre-fetch tools with offload → budget facts →
      L3 compaction → LLM generate → optional read_memory round.
    """
    trace: list[str] = []
    usage = TokenUsage()
    gs = state if state is not None else GenericState()

    gs.turn_count += 1

    # ── L1 — update structured memory (LLM-driven: bookings + catalog) ───
    prior_assistant_reply = ""
    for msg in reversed(history or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            prior_assistant_reply = str(msg.get("content") or "")
            break
    pivot_fragments = run_extractor(
        gs, user_message, prior_assistant_reply, client, trace
    )

    # ── Rule-based routing + L2 offload ──────────────────────────────────
    pre_calls = route_and_call(user_message)
    for tc in pre_calls:
        log = f"tool call → {tc.name}({json.dumps(tc.args, ensure_ascii=False)[:80]})"
        trace.append(log)
        if tc.offload_path:
            trace.append(
                f"tool result OFFLOADED ({tc.raw_tokens:,} tok → disk: {tc.offload_path})"
            )
        else:
            preview = tc.result[:200] + ("…" if len(tc.result) > 200 else "")
            trace.append(f"tool result inline ({len(tc.result):,} chars):\n{preview}")
        if verbose:
            print(f"  {log}", flush=True)

    # ── Deterministic budget facts (pinned ahead of tool results) ────────
    context_block = format_context_block(pre_calls)
    budget_facts = build_budget_facts_block(gs, user_message)
    if budget_facts:
        trace.append("budget facts injected (Python-computed — see context block)")
        context_block = (
            f"{budget_facts}\n\n{context_block}" if context_block else budget_facts
        )

    augmented_message = user_message
    if context_block:
        augmented_message = f"{context_block}\n\nUser query: {user_message}"

    # ── Build message list ───────────────────────────────────────────────
    pinned = gs.to_prompt_block()
    pinned_tokens = count_text_tokens(pinned) if pinned else 0
    base_system = _system_prompt_for_model()
    system_content = base_system + (f"\n\n{pinned}" if pinned else "")
    if ENABLE_THINKING:
        trace.append("thinking: ENABLED (enable_thinking=True, /no_think stripped)")
    messages: list[dict] = [{"role": "system", "content": system_content}]
    # Compress prior turns before handing history to the model. By default
    # (HISTORY_MODE=summary) ONE LLM call folds every past user+assistant turn
    # into a single <history_summary> block — so on turn 4 the model sees one
    # summary covering turns 1-3 instead of 6 raw messages. The Gradio UI
    # still shows every full reply to the user; only the model view is
    # compressed.
    messages.extend(_prepare_history_for_model(history, client, trace))
    messages.append({"role": "user", "content": augmented_message})

    # ── L3 — conversation compaction ──────────────────────────────────────
    compaction_event: Optional[CompactionEvent] = None
    if should_compact(messages):
        gs.compaction_count += 1
        messages, compaction_event = compact(
            messages,
            gs,
            client,
            thread_id=gs.thread_id,
            pivot_fragments=pivot_fragments,
            compaction_idx=gs.compaction_count,
        )
        if compaction_event.fired:
            trace.append(
                f"compaction fired: archived {compaction_event.turns_archived} message(s) "
                f"({compaction_event.tokens_before:,} → {compaction_event.tokens_after:,} tok) "
                f"→ {compaction_event.archive_path}"
            )
            if verbose:
                print(
                    f"  ↳ compaction: {compaction_event.tokens_before:,} → "
                    f"{compaction_event.tokens_after:,} tok "
                    f"(saved {compaction_event.saved_tokens():,})",
                    flush=True,
                )

    trace.append("── LLM call ──────────────────────")
    _dump_prompt(messages, gs.turn_count, trace)

    llm_prompt_rounds: list[list[dict[str, str]]] = []

    # Only expose tools + auto-choice when there's actually an offloaded payload
    # the model might want to fetch back via read_memory. On turns with no
    # <TRUNCATED> marker, passing tools=TOOL_SCHEMAS + tool_choice="auto" causes
    # small models (SmolLM3-3B) to slip into document-completion mode and parrot
    # the user message instead of answering.
    has_offload_marker = "<TRUNCATED>" in augmented_message
    tool_kwargs: dict[str, Any] = (
        {"tools": TOOL_SCHEMAS, "tool_choice": "auto"}
        if has_offload_marker
        else {"tool_choice": "none"}
    )
    trace.append(
        f"tool_choice={tool_kwargs['tool_choice']} "
        f"(offload marker present: {has_offload_marker})"
    )

    # ── First LLM call ───────────────────────────────────────────────────
    llm_prompt_rounds.append(_snapshot_messages(messages))
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        **tool_kwargs,
        **_thinking_extra_body(),
    )

    msg = response.choices[0].message
    raw_usage = response.usage
    read_memory_calls: list[dict] = []

    follow_up = _handle_tool_calls(msg, trace, read_memory_calls)

    # ── If read_memory was emitted, give the model one more shot ─────────
    if follow_up:
        messages = messages + follow_up
        llm_prompt_rounds.append(_snapshot_messages(messages))
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            tools=TOOL_SCHEMAS,
            tool_choice="none",   # no further tool use this turn
            **_thinking_extra_body(),
        )
        msg = response.choices[0].message
        raw_usage = response.usage

    # ── Plain-text read_memory fallback (SmolLM3 quirk) ──────────────────
    raw_content = msg.content or ""
    if not follow_up:
        fallback = _handle_plaintext_read_memory(
            raw_content, trace, read_memory_calls
        )
        if fallback is not None:
            messages = messages + [fallback]
            llm_prompt_rounds.append(_snapshot_messages(messages))
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.3,
                tools=TOOL_SCHEMAS,
                tool_choice="none",
                **_thinking_extra_body(),
            )
            msg = response.choices[0].message
            raw_usage = response.usage
            raw_content = msg.content or ""

    if raw_usage:
        usage = TokenUsage(
            prompt=raw_usage.prompt_tokens,
            completion=raw_usage.completion_tokens,
            total=raw_usage.total_tokens,
            context_pct=raw_usage.total_tokens / CONTEXT_LIMIT * 100,
        )
        trace.append(_token_bar(usage))
        if verbose:
            print(_token_bar(usage), flush=True)

    clean_reply, think_text = _strip_think(raw_content)
    if think_text:
        short = think_text[:400] + ("…" if len(think_text) > 400 else "")
        trace.append(f"thinking:\n{short}")

    # Belt-and-suspenders: kill any naked tool-call JSON the model may have
    # emitted as plain text. The rule-based router has already run the tools
    # the user needed, so leaked JSON blobs are always noise.
    clean_reply, stripped = _strip_tool_call_leak(clean_reply)
    if stripped:
        trace.append(
            f"tool-call leak stripped from reply: {stripped} blob(s) removed"
        )

    return AgentReply(
        text=clean_reply or "(no reply)",
        usage=usage,
        trace=trace,
        tool_calls=pre_calls,
        augmented_message=augmented_message,
        state=gs,
        pinned_block_tokens=pinned_tokens,
        compaction=compaction_event,
        read_memory_calls=read_memory_calls,
        llm_prompt_rounds=llm_prompt_rounds,
    )


# ---------------------------------------------------------------------------
# Gradio UI entry-point (UI lives in ui.py — imported from there)
# ---------------------------------------------------------------------------
def chat_fn(
    message: str,
    history: list[dict],
    state: GenericState | None = None,
) -> tuple[str, str, GenericState]:
    """Adapter for `gr.ChatInterface` (optional external GenericState)."""
    gs = state if state is not None else GenericState()
    reply = run_agent(message, history, state=gs, verbose=False)
    return reply.text, "\n".join(reply.trace), gs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1 Travel Agent (no UI)")
    parser.add_argument("message", nargs="?", default="Find hotels in Paris under $300/night")
    args = parser.parse_args()

    print(f"Sending: {args.message!r}")
    result = run_agent(args.message, [], verbose=True)
    print("\n=== Reply ===")
    print(result.text)

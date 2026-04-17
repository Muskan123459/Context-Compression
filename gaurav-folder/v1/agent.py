"""
v2 — Bloat-and-Break agent + L1 sticky GlobalState.

Same bloated tools and full history as v1, plus a bounded GlobalState block
pinned into the system prompt each turn (regex + optional LLM extraction).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from openai import OpenAI

from config import (
    CONTEXT_LIMIT,
    MAX_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    VLLM_BASE_URL,
    WARN_THRESHOLD,
)
from global_state import GlobalState, count_text_tokens
from state_extractor import maybe_llm_extract, merge_regex
from tool_router import ToolCall, format_context_block, route_and_call

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
    augmented_message:    str             = ""   # user msg + injected tool blobs (store in history)
    state:                GlobalState     = field(default_factory=GlobalState)
    pinned_block_tokens:  int             = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> tuple[str, str]:
    """Split `<think>…</think>` blocks from the visible reply.

    Handles a truncated think block (opening `<think>` with no closer — happens
    when the model runs out of `max_tokens` mid-reasoning) by treating
    everything from the opening tag onward as hidden thought.
    """
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
# Core agentic loop
# ---------------------------------------------------------------------------
def run_agent(
    user_message: str,
    history: list[dict],
    *,
    state: GlobalState | None = None,
    verbose: bool = True,
) -> AgentReply:
    """
    Pre-fetch tool data via rule-based routing, inject into context,
    then ask the LLM to generate a text reply.

    The LLM never has to emit tool_call JSON — it just reads the pre-fetched
    results and writes a response.  This is reliable with any model regardless
    of function-calling capability.

    Parameters
    ----------
    user_message : str
    history      : OpenAI-format message list (role/content dicts), WITHOUT system
    verbose      : if True, prints token bar + trace to stdout

    Returns
    -------
    AgentReply with .text, .usage, .trace, .tool_calls, .state, .pinned_block_tokens
    """
    trace: list[str] = []
    usage = TokenUsage()
    gs = state if state is not None else GlobalState()

    gs.turn_count += 1
    merge_regex(gs, user_message)
    maybe_llm_extract(gs, user_message, client, trace)

    # ── Step 1: Rule-based tool routing ──────────────────────────────────
    pre_calls = route_and_call(user_message)

    for tc in pre_calls:
        preview = tc.result[:200] + ("…" if len(tc.result) > 200 else "")
        log = f"tool call → {tc.name}({json.dumps(tc.args, ensure_ascii=False)[:80]})"
        trace.append(log)
        trace.append(f"tool result ({len(tc.result):,} chars):\n{preview}")
        if verbose:
            print(f"  {log}", flush=True)

    # ── Step 2: Build context block from tool results ─────────────────────
    context_block = format_context_block(pre_calls)

    # Augment the user message with the tool context (invisible to end-user)
    augmented_message = user_message
    if context_block:
        augmented_message = f"{context_block}\n\nUser query: {user_message}"

    # ── Step 3: Build message list and call LLM ───────────────────────────
    pinned = gs.to_prompt_block()
    pinned_tokens = count_text_tokens(pinned) if pinned else 0
    system_content = SYSTEM_PROMPT + (f"\n\n{pinned}" if pinned else "")
    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": augmented_message})

    trace.append("── LLM call ──────────────────────")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )

    msg      = response.choices[0].message
    raw_usage = response.usage
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

    raw_content = msg.content or ""
    _, think_text = _strip_think(raw_content)
    if think_text:
        short = think_text[:400] + ("…" if len(think_text) > 400 else "")
        trace.append(f"thinking:\n{short}")

    clean_reply, _ = _strip_think(raw_content)

    return AgentReply(
        text=clean_reply or "(no reply)",
        usage=usage,
        trace=trace,
        tool_calls=pre_calls,
        augmented_message=augmented_message,
        state=gs,
        pinned_block_tokens=pinned_tokens,
    )


# ---------------------------------------------------------------------------
# Gradio UI entry-point  (UI lives in ui.py — imported from there)
# ---------------------------------------------------------------------------
def chat_fn(
    message: str,
    history: list[dict],
    state: GlobalState | None = None,
) -> tuple[str, str, GlobalState]:
    """Adapter for gr.ChatInterface (optional external GlobalState)."""
    gs = state if state is not None else GlobalState()
    reply = run_agent(message, history, state=gs, verbose=False)
    return reply.text, "\n".join(reply.trace), gs


if __name__ == "__main__":
    # Quick sanity-check without Gradio
    import argparse

    parser = argparse.ArgumentParser(description="v1 Travel Agent (no UI)")
    parser.add_argument("message", nargs="?", default="Find hotels in Paris under $300/night")
    args = parser.parse_args()

    print(f"Sending: {args.message!r}")
    result = run_agent(args.message, [], verbose=True)
    print("\n=== Reply ===")
    print(result.text)

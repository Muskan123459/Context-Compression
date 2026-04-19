#!/usr/bin/env python3
"""
Gradio UI for the 3-layer compression pipeline.

Side panels:
  1. Live GenericState JSON (pinned into the system prompt each turn)
  2. Token bar + numbers for the last turn
  3. Tool calls routed (L2 offload path + raw size)
  4. Offloaded tool dumps on disk (memory_store/) + model input (exact messages sent to vLLM)
  5. Compaction events across the session
  6. Full agent trace
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import gradio as gr

from agent import run_agent
from compaction import CompactionEvent
from config import (
    CONTEXT_LIMIT,
    CONVERSATION_STORE_DIR,
    MEMORY_STORE_DIR,
    MODEL,
    PERSISTENT_STATE_PATH,
    VLLM_BASE_URL,
    WARN_THRESHOLD,
)
from global_state import GenericState, count_text_tokens
from tool_router import ToolCall


# ---------------------------------------------------------------------------
# UI session state — tracks everything we want to show cross-turn
# ---------------------------------------------------------------------------
def _load_persistent_state() -> GenericState:
    return GenericState.load(PERSISTENT_STATE_PATH)


@dataclass
class UISession:
    state: GenericState = field(default_factory=_load_persistent_state)
    offloaded: List[dict] = field(default_factory=list)
    compactions: List[CompactionEvent] = field(default_factory=list)
    # Sum of API-reported usage across successful chat turns (main reply LLM only).
    cumulative_prompt: int = 0
    cumulative_completion: int = 0
    cumulative_total: int = 0


def _refresh_state_panel(session: "UISession") -> tuple["UISession", dict]:
    """Re-read persistent memory from disk on page load so the JSON panel is
    populated before the user sends anything."""
    session.state = GenericState.load(PERSISTENT_STATE_PATH)
    return session, session.state.to_dict()


def _clear_memory(session: "UISession") -> tuple["UISession", dict]:
    """Wipe the on-disk persistent memory and reset this session's state."""
    try:
        PERSISTENT_STATE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    session.state = GenericState()
    session.offloaded.clear()
    session.compactions.clear()
    session.cumulative_prompt = 0
    session.cumulative_completion = 0
    session.cumulative_total = 0
    return session, session.state.to_dict()


# ---------------------------------------------------------------------------
# Side-panel renderers
# ---------------------------------------------------------------------------
def _token_panel_html(
    prompt: int,
    completion: int,
    total: int,
    *,
    session_prompt: int = 0,
    session_completion: int = 0,
    session_total: int = 0,
) -> str:
    pct  = (total / CONTEXT_LIMIT * 100) if CONTEXT_LIMIT else 0
    fill = min(pct, 100.0)

    if pct >= WARN_THRESHOLD * 100:
        colour, label = "#dc2626", "NEAR LIMIT"
    elif pct >= 50:
        colour, label = "#f59e0b", "bloat visible"
    else:
        colour, label = "#10b981", "comfortable"

    return f"""
<div style="font-family: ui-monospace, Menlo, monospace; font-size: 0.82rem;">
  <div style="color:#64748b; font-size:0.75rem; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.04em;">Last turn</div>
  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
    <span><b>{total:,}</b> / {CONTEXT_LIMIT:,} tokens
          <span style="color:#64748b;">({pct:.1f}%)</span></span>
    <span style="color:{colour}; font-weight:600;">{label}</span>
  </div>
  <div style="background:#e2e8f0; border-radius:6px; height:12px; overflow:hidden;">
    <div style="width:{fill}%; background:{colour}; height:100%;"></div>
  </div>
  <div style="margin-top:6px; color:#475569;">
    prompt: {prompt:,} &nbsp;·&nbsp; completion: {completion:,}
  </div>
  <hr style="border:none; border-top:1px solid #e2e8f0; margin:12px 0 0;" />
  <div style="color:#64748b; font-size:0.75rem; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.04em;">Session total</div>
  <div style="font-size:1.45rem; font-weight:700; color:#0f172a; line-height:1.2;">
    {session_total:,} <span style="font-size:0.82rem; font-weight:500; color:#64748b;">tokens</span>
  </div>
  <div style="margin-top:8px; color:#475569; font-size:0.78rem;">
    prompt: {session_prompt:,} &nbsp;·&nbsp; completion: {session_completion:,}
  </div>
</div>
""".strip()


def _tool_calls_panel(calls: list[ToolCall]) -> str:
    if not calls:
        return (
            "_No tools routed this turn._\n\n"
            "The rule-based router decided this message didn't need any tools "
            "— usually because it's a booking confirmation or pure conversation."
        )

    lines = [f"**{len(calls)} tool call(s)** fired by the rule-based router:\n"]
    for i, tc in enumerate(calls, 1):
        args_str = json.dumps(tc.args, ensure_ascii=False)
        if len(args_str) > 140:
            args_str = args_str[:140] + "…"
        lines.append(f"**{i}. `{tc.name}`**")
        lines.append(f"   • args: `{args_str}`")
        if tc.offload_path:
            lines.append(
                f"   • result: **offloaded** ({tc.raw_tokens:,} tok → "
                f"`{tc.offload_path}`)"
            )
        else:
            lines.append(f"   • result: inline ({tc.raw_tokens:,} tok)")
        lines.append("")
    return "\n".join(lines)


def _offloaded_rows() -> list[list]:
    """Snapshot of `memory_store/` contents for the Dataframe."""
    if not MEMORY_STORE_DIR.exists():
        return []
    rows: list[list] = []
    for p in sorted(MEMORY_STORE_DIR.glob("*.json")):
        try:
            head = p.read_text(encoding="utf-8")[:160].replace("\n", " ")
            size = p.stat().st_size
        except Exception:
            head, size = "(unreadable)", 0
        tool = p.name.split("_", 1)[0]
        rows.append([tool, p.name, f"{size:,} B", head])
    return rows


def _compaction_log_text(events: list[CompactionEvent]) -> str:
    if not events:
        return "_No compaction events yet._"
    lines = [f"**{len(events)} compaction event(s) this session:**\n"]
    for i, ev in enumerate(events, 1):
        lines.append(
            f"**#{i}** archived {ev.turns_archived} msg(s) "
            f"→ `{ev.archive_path}` "
            f"({ev.tokens_before:,} → {ev.tokens_after:,} tok, "
            f"saved {ev.saved_tokens():,})"
        )
        if ev.pivot_fragments:
            lines.append(
                f"   • pivot fragments applied: "
                f"`{json.dumps(ev.pivot_fragments, ensure_ascii=False)}`"
            )
        if ev.summary:
            preview = ev.summary[:260] + ("…" if len(ev.summary) > 260 else "")
            lines.append(f"   • summary: {preview}")
        lines.append("")
    return "\n".join(lines)


def _format_llm_prompt_rounds(
    rounds: list[list[dict[str, str]]],
    *,
    max_total_chars: int = 200_000,
    max_msg_chars: int = 32_000,
) -> str:
    """Human-readable view of messages sent to vLLM (system + history + user)."""
    if not rounds:
        return "_No LLM prompt snapshot for this turn._"

    lines: list[str] = [
        "What the model receives:\n"
        "  • Message 1 (role=system): instructions + pinned global state\n"
        "  • Middle messages: prior conversation (shape depends on HISTORY_MODE: raw, "
        "distilled one-liners, or a single <history_summary> block)\n"
        "  • Last message (role=user): budget facts + tool results (if any) + your query\n"
        "Multiple 'LLM request' blocks below = multiple vLLM calls this turn "
        "(e.g. after read_memory).\n",
    ]
    total_written = 0
    for ri, snap in enumerate(rounds, start=1):
        header = f"\n{'═' * 72}\nLLM request {ri} / {len(rounds)}  —  {len(snap)} message(s)\n{'═' * 72}\n"
        chunk = header
        if total_written + len(chunk) > max_total_chars:
            lines.append("\n… [truncated: overall view limit reached]\n")
            break
        lines.append(header)
        total_written += len(header)

        for mi, msg in enumerate(snap):
            role = msg.get("role", "?")
            body = msg.get("content") or ""
            truncated = ""
            if len(body) > max_msg_chars:
                body = body[:max_msg_chars] + "\n… [message truncated for UI]"
                truncated = " (truncated)"
            tok = count_text_tokens(body)
            sub = (
                f"\n── Message {mi + 1} · role={role} · ~{tok:,} tok{truncated} ──\n{body}\n"
            )
            if total_written + len(sub) > max_total_chars:
                lines.append("\n… [truncated: overall view limit reached]\n")
                total_written = max_total_chars
                break
            lines.append(sub)
            total_written += len(sub)
        if total_written >= max_total_chars:
            break

    return "".join(lines)


# ---------------------------------------------------------------------------
# Gradio adapter
# ---------------------------------------------------------------------------
def chat(
    message: str,
    history: list[dict],
    session: UISession,
):
    try:
        print(f"\n━━━ Turn {session.state.turn_count + 1} ━━━ user: {message[:100]!r}",
              flush=True)
        reply = run_agent(message, history, state=session.state, verbose=True)
        for tc in reply.tool_calls:
            extra = f"  [offloaded: {tc.offload_path}]" if tc.offload_path else ""
            print(
                f"  ↳ tool: {tc.name}({tc.args})  [{tc.raw_tokens:,} tok]{extra}",
                flush=True,
            )
        if reply.compaction and reply.compaction.fired:
            print(
                f"  ↳ compaction #{session.state.compaction_count}: "
                f"{reply.compaction.tokens_before:,} → "
                f"{reply.compaction.tokens_after:,} tok",
                flush=True,
            )
        print(f"  ↳ reply head: {reply.text[:160]!r}", flush=True)
    except Exception as exc:
        err = f"**ERROR** while calling vLLM: `{type(exc).__name__}: {exc}`"
        return (
            err,
            err,
            _token_panel_html(
                0,
                0,
                0,
                session_prompt=session.cumulative_prompt,
                session_completion=session.cumulative_completion,
                session_total=session.cumulative_total,
            ),
            "_Agent crashed._",
            session.state.to_dict(),
            _offloaded_rows(),
            "_No LLM prompt snapshot._",
            _compaction_log_text(session.compactions),
            session,
        )

    # Record observability artefacts
    for tc in reply.tool_calls:
        if tc.offload_path and not any(
            r["path"] == tc.offload_path for r in session.offloaded
        ):
            session.offloaded.append(
                {
                    "path": tc.offload_path,
                    "tool": tc.name,
                    "raw_tokens": tc.raw_tokens,
                    "raw_bytes": tc.raw_bytes,
                }
            )
    if reply.compaction and reply.compaction.fired:
        session.compactions.append(reply.compaction)

    session.state.save(PERSISTENT_STATE_PATH)

    session.cumulative_prompt += reply.usage.prompt
    session.cumulative_completion += reply.usage.completion
    session.cumulative_total += reply.usage.total

    trace_text = "\n".join(reply.trace)
    token_html = _token_panel_html(
        reply.usage.prompt,
        reply.usage.completion,
        reply.usage.total,
        session_prompt=session.cumulative_prompt,
        session_completion=session.cumulative_completion,
        session_total=session.cumulative_total,
    )
    print(
        f"  [session] cumulative: {session.cumulative_total:,} tok "
        f"(prompt sum {session.cumulative_prompt:,}, completion sum {session.cumulative_completion:,})",
        flush=True,
    )
    tools_md = _tool_calls_panel(reply.tool_calls)
    offload_rows = _offloaded_rows()
    model_input_text = _format_llm_prompt_rounds(reply.llm_prompt_rounds)
    compaction_md = _compaction_log_text(session.compactions)

    return (
        reply.text,
        trace_text,
        token_html,
        tools_md,
        reply.state.to_dict(),
        offload_rows,
        model_input_text,
        compaction_md,
        session,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1440px !important; margin: 0 auto; }
#trace-box textarea { font-family: ui-monospace, Menlo, monospace; font-size: 0.76rem; }
#state-box { max-height: 320px; overflow: auto; }
#offload-box table { font-size: 0.72rem; font-family: ui-monospace, Menlo, monospace; }
#model-input-box textarea { font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Travel Planner — 3-layer compression") as demo:
        gr.Markdown(
            f"## Travel Planner — 3-layer compression pipeline\n"
            f"L1 GenericState (hybrid schema, pivot-aware) · "
            f"L2 tool-output offload to `memory_store/` · "
            f"L3 auto-compaction of old turns → `conversation_history/`\n\n"
            f"<sub>Model: `{MODEL}` &nbsp;·&nbsp; endpoint: `{VLLM_BASE_URL}` "
            f"&nbsp;·&nbsp; context limit: **{CONTEXT_LIMIT:,}** tokens</sub>"
        )

        session = gr.State(UISession())

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### Global state (L1 — pinned in prompt)")
                state_box = gr.JSON(value={}, elem_id="state-box")
                clear_btn = gr.Button(
                    "Delete memory (start from scratch)",
                    variant="stop",
                    size="sm",
                )

                gr.Markdown("#### Token usage")
                token_box = gr.HTML(
                    value=_token_panel_html(0, 0, 0),
                    elem_id="token-box",
                )

                gr.Markdown("#### Tools routed (last turn)")
                tools_box = gr.Markdown(
                    value="_Send a message to see which tools fire._",
                    elem_id="tools-box",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Offloaded tool dumps (L2 — `memory_store/`)")
                        offload_box = gr.Dataframe(
                            headers=["tool", "file", "size", "preview"],
                            datatype=["str", "str", "str", "str"],
                            value=_offloaded_rows(),
                            interactive=False,
                            wrap=True,
                            elem_id="offload-box",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model input (what the LLM sees this turn)")
                        model_input_box = gr.Textbox(
                            show_label=False,
                            lines=16,
                            max_lines=24,
                            interactive=False,
                            elem_id="model-input-box",
                            placeholder=(
                                "After you send a message: system prompt + pinned state, "
                                "conversation memory, and the augmented user message appear here."
                            ),
                        )

                gr.Markdown("#### Compaction events (L3 — this session)")
                compaction_box = gr.Markdown(
                    value="_No compaction events yet._",
                    elem_id="compaction-box",
                )

                gr.Markdown("#### Agent trace")
                trace_box = gr.Textbox(
                    show_label=False,
                    lines=18,
                    max_lines=18,
                    interactive=False,
                    placeholder="Tool calls, token counts, thinking, compactions…",
                    elem_id="trace-box",
                )

            with gr.Column(scale=5):
                gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[session],
                    additional_outputs=[
                        trace_box,
                        token_box,
                        tools_box,
                        state_box,
                        offload_box,
                        model_input_box,
                        compaction_box,
                        session,
                    ],
                    chatbot=gr.Chatbot(
                        height=620,
                        placeholder=(
                            "<b>Travel Planner — 3-layer compression</b><br>"
                            "Try: <i>Plan a multi-city trip: Paris, Tokyo, Bali, "
                            "total budget $3,800. I'm allergic to shellfish.</i>"
                        ),
                    ),
                    examples=[
                        ["I want to plan a 5-day trip to Tokyo and Kyoto. Budget is $3,000. I'm severely allergic to shellfish."],
                        ["Plan a beach vacation in Bali for next month."],
                        ["Actually, scratch Bali entirely — let's do Switzerland instead, I want mountains."],
                        ["Summarize my trip plan so far."],
                    ],
                    cache_examples=False,
                )

        clear_btn.click(fn=_clear_memory, inputs=[session], outputs=[session, state_box])
        demo.load(fn=_refresh_state_panel, inputs=[session], outputs=[session, state_box])

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Travel Agent — 3-layer compression UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public share link.")
    args = parser.parse_args()

    print(f"→ Connecting to vLLM at {VLLM_BASE_URL} (model: {MODEL})")
    print(f"→ Context limit:  {CONTEXT_LIMIT:,} tokens (warn at {WARN_THRESHOLD * 100:.0f}%)")
    print(f"→ memory_store:         {MEMORY_STORE_DIR}")
    print(f"→ conversation_history: {CONVERSATION_STORE_DIR}")

    MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSATION_STORE_DIR.mkdir(parents=True, exist_ok=True)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=_CSS,
    )


if __name__ == "__main__":
    main()

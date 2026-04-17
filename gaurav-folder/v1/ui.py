#!/usr/bin/env python3
"""
v2 — Gradio UI for the Bloat-and-Break agent + L1 GlobalState.

Run AFTER a vLLM server is up on VLLM_BASE_URL (default http://localhost:8000/v1):

    python v1/ui.py                    # launches on :7860
    python v1/ui.py --port 7861        # custom port
    python v1/ui.py --share            # public tunnel

Side panels:
  1. Live GlobalState (pinned into the system prompt each turn)
  2. Token bar + numbers for the last turn
  3. Tools routed + full agent trace
"""
from __future__ import annotations

import argparse
import json

import gradio as gr

from agent import run_agent
from config import CONTEXT_LIMIT, MODEL, VLLM_BASE_URL, WARN_THRESHOLD
from global_state import GlobalState
from tool_router import ToolCall


# ---------------------------------------------------------------------------
# Side-panel HTML/markdown renderers
# ---------------------------------------------------------------------------
def _token_panel_html(prompt: int, completion: int, total: int) -> str:
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
</div>
""".strip()


def _tool_calls_panel(calls: list[ToolCall]) -> str:
    if not calls:
        return (
            "_No tools routed this turn._\n\n"
            "The rule-based router decided this message didn't need any of "
            "`web_search` / `places_search` / `weather_fetch` / `budget_tracker` "
            "— usually because it's a booking confirmation or pure conversation."
        )

    lines = [f"**{len(calls)} tool call(s)** fired by the rule-based router:\n"]
    for i, tc in enumerate(calls, 1):
        args_str = json.dumps(tc.args, ensure_ascii=False)
        if len(args_str) > 140:
            args_str = args_str[:140] + "…"
        lines.append(f"**{i}. `{tc.name}`**")
        lines.append(f"   • args: `{args_str}`")
        lines.append(f"   • result: `{len(tc.result):,}` chars of JSON")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio adapter — returns (reply, trace, token_html, tools_md, state_dict, gs)
# ---------------------------------------------------------------------------
def chat(
    message: str,
    history: list[dict],
    gs: GlobalState,
) -> tuple[str, str, str, str, dict, GlobalState]:
    try:
        reply = run_agent(message, history, state=gs, verbose=False)
    except Exception as exc:
        err = f"**ERROR** while calling vLLM: `{type(exc).__name__}: {exc}`"
        return (
            err,
            err,
            _token_panel_html(0, 0, 0),
            "_Agent crashed._",
            gs.to_dict(),
            gs,
        )

    trace_text = "\n".join(reply.trace)
    token_html = _token_panel_html(
        reply.usage.prompt, reply.usage.completion, reply.usage.total
    )
    tools_md = _tool_calls_panel(reply.tool_calls)
    return reply.text, trace_text, token_html, tools_md, reply.state.to_dict(), gs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto; }
#trace-box textarea { font-family: ui-monospace, Menlo, monospace; font-size: 0.76rem; }
#state-box { max-height: 320px; overflow: auto; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Travel Planner — v2 (Sticky GlobalState)") as demo:
        gr.Markdown(
            f"## Travel Planner — v2 *(L1 sticky GlobalState)*\n"
            f"Same bloated tool fixtures as v1; **GlobalState** is extracted each turn "
            f"(regex + optional LLM) and pinned in the system prompt so budget and "
            f"preferences stay consistent.\n\n"
            f"<sub>Model: `{MODEL}` &nbsp;·&nbsp; endpoint: `{VLLM_BASE_URL}` "
            f"&nbsp;·&nbsp; context limit: **{CONTEXT_LIMIT:,}** tokens</sub>"
        )

        g_state = gr.State(GlobalState())

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### Global state (pinned in prompt)")
                state_box = gr.JSON(value={}, elem_id="state-box")

                gr.Markdown("#### Token usage (last turn)")
                token_box = gr.HTML(
                    value=_token_panel_html(0, 0, 0),
                    elem_id="token-box",
                )

                gr.Markdown("#### Tools routed (last turn)")
                tools_box = gr.Markdown(
                    value="_Send a message to see which tools fire._",
                    elem_id="tools-box",
                )

                gr.Markdown("#### Agent trace")
                trace_box = gr.Textbox(
                    show_label=False,
                    lines=22,
                    max_lines=22,
                    interactive=False,
                    placeholder="Tool calls, token counts, and thinking appear here…",
                    elem_id="trace-box",
                )

            with gr.Column(scale=5):
                gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[g_state],
                    additional_outputs=[trace_box, token_box, tools_box, state_box, g_state],
                    chatbot=gr.Chatbot(
                        height=560,
                        placeholder=(
                            "<b>Travel Planner — v2</b><br>"
                            "Try: <i>Plan a multi-city trip: Paris, Tokyo, Bali, "
                            "total budget $3,800</i>"
                        ),
                    ),
                    examples=[
                        ["Find hotels in Paris under $250/night"],
                        ["What's the weather like in Tokyo next week?"],
                        ["Plan a 3-night Bali stay — Ubud, breakfast included"],
                        ["I have a $3,800 budget for Paris + Tokyo + Bali — what do you suggest?"],
                    ],
                    cache_examples=False,
                )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Travel Agent — Gradio UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public share link.")
    args = parser.parse_args()

    print(f"→ Connecting to vLLM at {VLLM_BASE_URL} (model: {MODEL})")
    print(f"→ Context limit:  {CONTEXT_LIMIT:,} tokens "
          f"(warn at {WARN_THRESHOLD * 100:.0f}%)")

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

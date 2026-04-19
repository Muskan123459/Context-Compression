#!/usr/bin/env python3
"""
v0 — Travel agent with Qwen on vLLM (OpenAI-compatible API) + four mock tools
+ Gradio UI. The Message box clears after each send; the Agent activity log
accumulates every turn (payload preview, tool/LLM trace, assistant reply).

Match vLLM --max-model-len to CONTEXT_LIMIT (default 16k).

Start vLLM first: see gaurav-folder/start.sh
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

# Allow `from tools import …` when running as `python agent.py` from any cwd.
_V0_ROOT = Path(__file__).resolve().parent
if str(_V0_ROOT) not in sys.path:
    sys.path.insert(0, str(_V0_ROOT))

from tools import TOOL_SCHEMAS, dispatch_tool  # noqa: E402

# ---------------------------------------------------------------------------
# Config — Qwen + 16k window (must match vLLM --max-model-len)
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT", "16384"))
MAX_TOOL_ROUNDS = 5  # safety ceiling; stops infinite tool loops

client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

TOOLS = TOOL_SCHEMAS

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a travel concierge. You help users plan trips.

You have tools: web_search (flights / travel info), places_search (hotels, restaurants, attractions), weather_fetch, budget_tracker (set_budget, add_expense, get_summary).

RULES:
- When the user asks about hotels, restaurants, attractions, flights, weather, or trip budgets, call the right tool(s). Do not invent listings or fares from memory.
- You may call multiple tools in one turn if needed.
- After tool results, write a concise, helpful reply that cites the actual results.
- Never describe what you are going to do. Just do it.
- Keep replies practical and skimmable."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> tuple[str, str]:
    think_match = re.search(
        r"<think>(.*?)</think>", text, flags=re.DOTALL
    )
    think_text = think_match.group(1).strip() if think_match else ""
    clean = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    ).strip()
    return clean, think_text


def _dispatch_tool(name: str, args: dict) -> str:
    return dispatch_tool(name, args)


def _coerce_content_to_text(content: Any) -> str:
    """Gradio Chatbot may store content as str or as multimodal parts [{'text': '...', 'type': 'text'}]."""
    if content is None:
        return ""
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
    return str(content)


def _format_payload_for_log(messages: list[dict], *, max_chars_per_msg: int = 8000) -> str:
    """Human-readable view of system + user + assistant messages (no tool rounds)."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        content = m.get("content")
        if content is None:
            continue
        text = _coerce_content_to_text(content)
        if len(text) > max_chars_per_msg:
            text = text[:max_chars_per_msg] + "\n… [truncated]"
        lines.append(f"[{role}]\n{text}")
    return "\n\n".join(lines) if lines else "(no text messages)"


# ---------------------------------------------------------------------------
# Agentic loop — returns (reply, trace_lines, turn_usage)
# turn_usage = (prompt_tokens, completion_tokens, total_tokens) summed over
# every chat.completions call in this turn (tool loops = multiple calls).
# ---------------------------------------------------------------------------
def run_agent(user_message: str, history: list[dict]) -> tuple[str, list[str], tuple[int, int, int]]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": _coerce_content_to_text(user_message)})
    trace: list[str] = [
        "── First vLLM request — text passed in (system prompt + prior chat + this user) ──",
        _format_payload_for_log(messages),
        "",
        "── Tool / LLM rounds (this turn) ──",
    ]
    turn_prompt = 0
    turn_completion = 0
    turn_total = 0

    for round_i in range(MAX_TOOL_ROUNDS + 1):
        trace.append(f"── LLM call #{round_i + 1} ──────────────────────")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        msg = response.choices[0].message
        usage = response.usage
        if usage:
            pt = int(usage.prompt_tokens or 0)
            ct = int(usage.completion_tokens or 0)
            tt = int(usage.total_tokens or 0)
            turn_prompt += pt
            turn_completion += ct
            turn_total += tt
            trace.append(
                f"tokens  prompt={pt}  "
                f"completion={ct}  "
                f"total={tt}  "
                f"(context cap {CONTEXT_LIMIT:,})"
            )

        raw_content = msg.content or ""
        _, think_text = _strip_think(raw_content)
        if think_text:
            short_think = think_text[:400] + ("…" if len(think_text) > 400 else "")
            trace.append(f"thinking:\n{short_think}")

        if not msg.tool_calls:
            clean_reply, _ = _strip_think(raw_content)
            trace.append("→ no tool call, returning reply")
            trace.append(
                f"── This turn subtotal (all vLLM calls above): "
                f"prompt {turn_prompt:,} + completion {turn_completion:,} = {turn_total:,} total"
            )
            return clean_reply or "(no reply)", trace, (turn_prompt, turn_completion, turn_total)

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            args_str = json.dumps(fn_args, ensure_ascii=False)
            trace.append(f"tool call → {fn_name}({args_str})")

            result_str = _dispatch_tool(fn_name, fn_args)
            result_preview = result_str[:300] + ("…" if len(result_str) > 300 else "")
            trace.append(f"tool result:\n{result_preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    trace.append(
        f"── This turn subtotal (all vLLM calls above): "
        f"prompt {turn_prompt:,} + completion {turn_completion:,} = {turn_total:,} total"
    )
    return "(agent hit tool-call limit without a final reply)", trace, (turn_prompt, turn_completion, turn_total)


# ---------------------------------------------------------------------------
# Gradio session — cumulative Agent activity log
# ---------------------------------------------------------------------------
@dataclass
class ChatSession:
    """Stores each completed turn for the left-hand activity panel."""

    turns: list[dict] = field(default_factory=list)
    # API usage since page load (sum of all vLLM chat.completions in this session).
    cumulative_prompt: int = 0
    cumulative_completion: int = 0
    cumulative_total: int = 0


def _gr_history_to_messages(history: list) -> list[dict]:
    """Normalize Gradio chat history to OpenAI-style user/assistant dicts for run_agent.

    Gradio 4+ Chatbot uses a flat list of ``{"role", "content"}`` messages.
    Legacy ``[user, assistant]`` pairs are still accepted for compatibility.
    """
    out: list[dict] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role"):
            role = str(item["role"])
            if role not in ("user", "assistant"):
                continue
            out.append({"role": role, "content": _coerce_content_to_text(item.get("content"))})
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            u, a = item[0], item[1]
            ut = _coerce_content_to_text(u)
            at = _coerce_content_to_text(a)
            if ut.strip():
                out.append({"role": "user", "content": ut})
            if at.strip():
                out.append({"role": "assistant", "content": at})
    return out


def _format_activity_log(session: ChatSession) -> str:
    """Full cumulative log: each turn = payload+tools trace, then assistant reply."""
    if not session.turns:
        return "_Agent activity log is empty — send a message to see the pipeline._"
    blocks: list[str] = []
    for i, t in enumerate(session.turns, 1):
        tr = str(t.get("trace", ""))
        ast = str(t.get("assistant", ""))
        blocks.append(
            f"{'=' * 56}\n"
            f" Turn {i}\n"
            f"{'=' * 56}\n\n"
            f"{tr}\n\n"
            f"── Assistant reply (chatbot) ──\n{ast}\n"
        )
    return "\n".join(blocks)


def _token_total_html(session: ChatSession) -> str:
    """One dominant cumulative total for the current browser session."""
    n = session.cumulative_total
    return f"""
<div class="v0-token-total" style="font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  padding: 1rem 1.25rem; background: #ffffff; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  color: #0f172a; border-radius: 12px; text-align: center; margin-bottom: 14px;
  border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 4px 20px rgba(15,23,42,0.04);">
  <div style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #64748b; margin-bottom: 8px;">Total tokens this session</div>
  <div style="font-size: 3.25rem; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em; color: #0f172a;">{n:,}</div>
  <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 10px;">Since page load · all turns · all vLLM calls</div>
</div>
""".strip()


def submit_chat(message: str, history: list, session: ChatSession):
    user_msg = (message or "").strip()
    if not user_msg:
        return history, "", _format_activity_log(session), _token_total_html(session), session

    hist = list(history or [])
    prior_messages = _gr_history_to_messages(hist)
    reply, trace_lines, (tp, tc, tt) = run_agent(user_msg, prior_messages)
    trace_text = "\n".join(trace_lines)

    session.cumulative_prompt += tp
    session.cumulative_completion += tc
    session.cumulative_total += tt

    # Gradio "messages" format: flat list of role/content dicts (see Chatbot._check_format).
    new_hist = hist + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]

    session.turns.append({
        "user": user_msg,
        "assistant": reply,
        "trace": trace_text,
    })
    activity = _format_activity_log(session)
    # Empty message box after each send so the next question is typed fresh.
    return new_hist, "", activity, _token_total_html(session), session


def clear_chat(session: ChatSession):
    session.turns.clear()
    session.cumulative_prompt = 0
    session.cumulative_completion = 0
    session.cumulative_total = 0
    return [], "", _format_activity_log(session), _token_total_html(session), session


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#trace-box textarea { font-family: monospace; font-size: 0.78rem; }
.v0-token-total { user-select: none; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Travel Planner (v0)") as demo:
        gr.Markdown(
            f"## Travel Planner (v0)\n"
            f"**Qwen** on vLLM · **{CONTEXT_LIMIT:,}-token** context cap · "
            f"four tools: `web_search`, `places_search`, `weather_fetch`, `budget_tracker`.  \n"
            f"Type in **Message** and send — the box clears after each reply. "
            f"**Agent activity log** (left) accumulates every turn: system prompt + chat history + your line, "
            f"then tool/LLM steps, then the assistant reply.\n\n"
            f"<sub>Model: `{MODEL}` &nbsp;·&nbsp; endpoint: `{VLLM_BASE_URL}`</sub>"
        )

        session = gr.State(ChatSession())

        with gr.Row():
            with gr.Column(scale=3):
                token_total = gr.HTML(
                    value=_token_total_html(ChatSession()),
                    elem_id="v0-token-total-box",
                )
                trace_box = gr.Textbox(
                    label="Agent activity log",
                    lines=28,
                    max_lines=28,
                    interactive=False,
                    placeholder="Cumulative: payload (system + history + user), tool/LLM trace, assistant reply per turn.",
                    elem_id="trace-box",
                )

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder=(
                        "<b>Travel Planner</b><br>"
                        "Try: <i>Plan Paris and Tokyo, budget $4000 — hotels and weather</i>"
                    ),
                )
                msg_box = gr.Textbox(
                    label="Message",
                    lines=6,
                    placeholder="Your question (clears after Send). Prior turns + tools appear in Agent activity log.",
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear chat", variant="secondary")

                gr.Examples(
                    examples=[
                        ["Plan a week: Paris then Tokyo, total budget $3,500. I want hotels and flight ideas."],
                        ["What's the weather in Bali in June and good restaurants?"],
                        ["Set budget to $2,000 and show a budget summary after suggesting Bali hotels."],
                    ],
                    inputs=[msg_box],
                    cache_examples=False,
                )

        send.click(
            submit_chat,
            inputs=[msg_box, chatbot, session],
            outputs=[chatbot, msg_box, trace_box, token_total, session],
        )
        msg_box.submit(
            submit_chat,
            inputs=[msg_box, chatbot, session],
            outputs=[chatbot, msg_box, trace_box, token_total, session],
        )
        clear_btn.click(
            clear_chat,
            inputs=[session],
            outputs=[chatbot, msg_box, trace_box, token_total, session],
        )

    return demo


if __name__ == "__main__":
    import argparse

    # Accept `--port7860` (no space) — common shell typo; becomes `--port 7860`.
    _norm: list[str] = []
    for _arg in sys.argv:
        _m = re.match(r"^--port(\d+)$", _arg)
        if _m:
            _norm.extend(["--port", _m.group(1)])
        else:
            _norm.append(_arg)
    sys.argv = _norm

    parser = argparse.ArgumentParser(
        description="v0 Travel Agent",
        epilog="Tip: use `--port 7860` (space) or `--port7860` (no space).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port (default 7860)")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Connecting to vLLM at {VLLM_BASE_URL} (model: {MODEL})")
    print(f"Context cap (set vLLM --max-model-len to match): {CONTEXT_LIMIT:,} tokens")
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=_CSS,
    )

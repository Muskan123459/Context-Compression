#!/usr/bin/env python3
"""
v0 — Hello Agent
SmolLM3-3B via vLLM (OpenAI-compatible endpoint) + one tool + Gradio UI.
Start vLLM first: see start.sh
"""
from __future__ import annotations

import json
import os
import re
import sys

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "HuggingFaceTB/SmolLM3-3B")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
MAX_TOOL_ROUNDS = 5  # safety ceiling; stops infinite tool loops

client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a travel concierge. You help users plan trips.

RULES:
- When a user asks about hotels, restaurants, or attractions, ALWAYS call places_search first. Do not answer from memory.
- After getting tool results, write a short, helpful reply referencing the actual results.
- Never describe what you are going to do. Just do it.
- Keep replies concise and practical."""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "places_search",
            "description": "Search for hotels, restaurants, or attractions in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris' or 'Tokyo'",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["hotels", "restaurants", "attractions"],
                        "description": "What to search for",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Hardcoded tool data (~200 tokens per city)
# ---------------------------------------------------------------------------
_PLACES_DB: dict[str, dict] = {
    "paris": {
        "hotels": [
            {"name": "Hôtel du Louvre", "stars": 4, "price_per_night": 220, "area": "1st arrondissement", "notes": "5-min walk to the Louvre"},
            {"name": "Le Marais Boutique", "stars": 3, "price_per_night": 145, "area": "3rd arrondissement", "notes": "Trendy neighbourhood, great cafes nearby"},
            {"name": "Grand Hôtel Opéra", "stars": 5, "price_per_night": 480, "area": "9th arrondissement", "notes": "Classic Haussmann building, rooftop bar"},
        ],
        "restaurants": [
            {"name": "Chez L'Ami Jean", "cuisine": "Basque", "price": "$$", "area": "7th arrondissement"},
            {"name": "Au Passage", "cuisine": "Natural wine bistro", "price": "$$", "area": "11th arrondissement"},
            {"name": "Septime", "cuisine": "Modern French", "price": "$$$", "area": "11th arrondissement", "notes": "Book weeks ahead"},
        ],
        "attractions": [
            {"name": "Musée d'Orsay", "type": "Museum", "entry": 16, "tip": "Buy timed tickets online"},
            {"name": "Sainte-Chapelle", "type": "Historic site", "entry": 13, "tip": "Come early, queues build fast"},
            {"name": "Père Lachaise Cemetery", "type": "Park/History", "entry": 0},
        ],
    },
    "tokyo": {
        "hotels": [
            {"name": "Shinjuku Granbell", "stars": 4, "price_per_night": 180, "area": "Shinjuku", "notes": "Designer rooms, near nightlife"},
            {"name": "Khaosan Tokyo Origami", "stars": 3, "price_per_night": 90, "area": "Asakusa", "notes": "Traditional neighbourhood feel"},
            {"name": "Park Hyatt Tokyo", "stars": 5, "price_per_night": 700, "area": "Shinjuku", "notes": "Lost in Translation hotel"},
        ],
        "restaurants": [
            {"name": "Ichiran Ramen", "cuisine": "Ramen", "price": "$", "area": "Multiple locations"},
            {"name": "Sushi Saito", "cuisine": "Omakase sushi", "price": "$$$$", "area": "Mita", "notes": "Reservation required months ahead"},
            {"name": "Gonpachi Nishi-Azabu", "cuisine": "Izakaya", "price": "$$", "area": "Nishi-Azabu"},
        ],
        "attractions": [
            {"name": "Senso-ji Temple", "type": "Temple", "entry": 0, "tip": "Go at dawn to beat crowds"},
            {"name": "TeamLab Borderless", "type": "Digital art", "entry": 32, "tip": "Book online"},
            {"name": "Tsukiji Outer Market", "type": "Market", "entry": 0, "tip": "Breakfast there is a must"},
        ],
    },
    "bali": {
        "hotels": [
            {"name": "Four Seasons Sayan", "stars": 5, "price_per_night": 900, "area": "Ubud", "notes": "Jungle-canopy pool"},
            {"name": "Katamama", "stars": 5, "price_per_night": 450, "area": "Seminyak", "notes": "Balinese craftsmanship throughout"},
            {"name": "Bisma Eight", "stars": 4, "price_per_night": 220, "area": "Ubud", "notes": "Rice terrace views"},
        ],
        "restaurants": [
            {"name": "Locavore", "cuisine": "Modern Indonesian", "price": "$$$", "area": "Ubud", "notes": "Best restaurant in Bali"},
            {"name": "Merah Putih", "cuisine": "Indonesian", "price": "$$", "area": "Seminyak"},
            {"name": "Naughty Nuri's", "cuisine": "BBQ ribs", "price": "$", "area": "Ubud"},
        ],
        "attractions": [
            {"name": "Tegallalang Rice Terraces", "type": "Nature", "entry": 2},
            {"name": "Tanah Lot Temple", "type": "Temple", "entry": 4, "tip": "Sunset is spectacular"},
            {"name": "Mount Batur Sunrise Trek", "type": "Adventure", "entry": 60, "tip": "Book a guide"},
        ],
    },
}

_DEFAULT_RESULT = {"note": "No data for that city yet. This is a demo with Paris, Tokyo, and Bali."}


def places_search(city: str, category: str = "hotels") -> dict:
    key = city.strip().lower()
    city_data = _PLACES_DB.get(key, {})
    if not city_data:
        return _DEFAULT_RESULT
    return {city: city_data.get(category, city_data)}


def _dispatch_tool(name: str, args: dict) -> str:
    if name == "places_search":
        result = places_search(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> tuple[str, str]:
    """
    Split model output into (visible_reply, think_block).
    Returns the cleaned reply and the raw thinking text (may be empty).
    """
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean, think_text


# ---------------------------------------------------------------------------
# Agentic loop — returns (reply, trace_lines)
# ---------------------------------------------------------------------------
def run_agent(user_message: str, history: list[dict]) -> tuple[str, list[str]]:
    """
    Returns:
      reply      — final assistant text (think tags stripped)
      trace      — list of log lines describing what happened
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    trace: list[str] = []

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
            trace.append(
                f"tokens  prompt={usage.prompt_tokens}  "
                f"completion={usage.completion_tokens}  "
                f"total={usage.total_tokens}"
            )

        # Surface thinking if present
        raw_content = msg.content or ""
        _, think_text = _strip_think(raw_content)
        if think_text:
            short_think = think_text[:400] + ("…" if len(think_text) > 400 else "")
            trace.append(f"thinking:\n{short_think}")

        # No tool call — final answer
        if not msg.tool_calls:
            clean_reply, _ = _strip_think(raw_content)
            trace.append("→ no tool call, returning reply")
            return clean_reply or "(no reply)", trace

        # Tool calls
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

            # Show a trimmed preview of the result
            result_preview = result_str[:300] + ("…" if len(result_str) > 300 else "")
            trace.append(f"tool result:\n{result_preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return "(agent hit tool-call limit without a final reply)", trace


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def chat(message: str, history: list[dict]) -> tuple[str, str]:
    reply, trace_lines = run_agent(message, history)
    trace_text = "\n".join(trace_lines)
    return reply, trace_text


_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

_CSS = """
footer { display: none !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#trace-box textarea { font-family: monospace; font-size: 0.78rem; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Travel Planner") as demo:
        gr.Markdown(
            f"## Travel Planner\n"
            f"Ask about hotels, restaurants, or attractions in **Paris, Tokyo, or Bali**.  \n"
            f"<sub>Model: `{MODEL}` &nbsp;·&nbsp; endpoint: `{VLLM_BASE_URL}` &nbsp;·&nbsp; v0 baseline — no compression</sub>"
        )

        with gr.Row():
            with gr.Column(scale=3):
                trace_box = gr.Textbox(
                    label="Agent Activity Log",
                    lines=28,
                    max_lines=28,
                    interactive=False,
                    placeholder="Tool calls, token counts, and thinking will appear here…",
                    elem_id="trace-box",
                )

            with gr.Column(scale=5):
                gr.ChatInterface(
                    fn=chat,
                    additional_outputs=[trace_box],
                    chatbot=gr.Chatbot(
                        height=500,
                        placeholder=(
                            "<b>Travel Planner</b><br>"
                            "Try: <i>find hotels in Paris</i> or <i>best restaurants in Tokyo</i>"
                        ),
                        avatar_images=(
                            None,
                            "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
                        ),
                    ),
                    examples=[
                        "Find hotels in Paris under $250/night",
                        "What are the best restaurants in Tokyo?",
                        "What should I visit in Bali?",
                        "Plan a 3-day trip to Paris — hotels and top attractions",
                    ],
                    cache_examples=False,
                )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v0 Travel Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Connecting to vLLM at {VLLM_BASE_URL} (model: {MODEL})")
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_THEME,
        css=_CSS,
    )

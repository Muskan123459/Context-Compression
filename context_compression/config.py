"""
v1 — shared configuration.

All tuneable values live here; everything else imports from this module.
"""
from __future__ import annotations

import os
from pathlib import Path

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# Default upgraded from SmolLM3-3B (too weak — leaked naked tool-call JSON,
# could not do multi-step budget arithmetic) to Qwen2.5-7B-Instruct-AWQ,
# which fits the A5000 24GB alongside a 16k KV cache and has native
# Hermes-style tool-calling.
MODEL: str = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1500"))
MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

# Model's hard context-window cap — must match the vLLM server's
# --max-model-len. Kept at 16k so both sides agree.
CONTEXT_LIMIT: int = int(os.getenv("CONTEXT_LIMIT", "16384"))

# Warn in the token counter once we exceed this fraction of CONTEXT_LIMIT
WARN_THRESHOLD: float = float(os.getenv("WARN_THRESHOLD", "0.75"))

# Pinned GenericState block: trim until under this many tokens (tiktoken / heuristic)
GLOBAL_STATE_TOKEN_CAP: int = int(os.getenv("GLOBAL_STATE_TOKEN_CAP", "300"))

# LLM call budgets for the domain-free structured-memory extractor
STATE_EXTRACTOR_MAX_TOKENS: int = int(os.getenv("STATE_EXTRACTOR_MAX_TOKENS", "220"))
PIVOT_CONFIRM_MAX_TOKENS: int = int(os.getenv("PIVOT_CONFIRM_MAX_TOKENS", "80"))

# ── Layer 2 — tool-output offload ────────────────────────────────────────────
# Any tool result whose serialised JSON exceeds this token count is written to
# `memory_store/` and replaced in chat history with a preview + `read_memory`
# pointer.
OFFLOAD_THRESHOLD_TOKENS: int = int(os.getenv("OFFLOAD_THRESHOLD_TOKENS", "1500"))

# Number of preview lines kept inline (the rest lives on disk).
OFFLOAD_PREVIEW_LINES: int = int(os.getenv("OFFLOAD_PREVIEW_LINES", "10"))

# Where offloaded tool dumps live. Resolved relative to the v1 package dir so
# the path is stable regardless of which working directory the agent is run from.
_V1_DIR = Path(__file__).resolve().parent
MEMORY_STORE_DIR: Path = Path(
    os.getenv("MEMORY_STORE_DIR", str(_V1_DIR / "memory_store"))
).resolve()

# Persistent Layer-1 memory (ChatGPT-style) — survives UI refreshes and new threads.
PERSISTENT_STATE_PATH: Path = Path(
    os.getenv("PERSISTENT_STATE_PATH", str(_V1_DIR / "persistent_state.json"))
).resolve()

# ── Layer 3 — conversation compaction ────────────────────────────────────────
# Trigger the compactor when total tokens exceed this fraction of CONTEXT_LIMIT.
COMPACT_THRESHOLD_PCT: float = float(os.getenv("COMPACT_THRESHOLD_PCT", "0.85"))

# Number of oldest user/assistant turn pairs to archive when compaction fires.
N_COMPACT_TURNS: int = int(os.getenv("N_COMPACT_TURNS", "4"))

# Max tokens the summariser may emit for the replacement `<summary>` block.
SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "250"))

# Per-turn history compression mode (how prior turns are shown to the model):
#   "summary"  — ONE LLM call folds all prior user+assistant turns into a
#                single <history_summary> block (default; requested behaviour).
#   "distill"  — each prior assistant reply is distilled to 1 line; user
#                messages pass through as-is (legacy behaviour).
#   "none"     — raw history; model sees every past turn verbatim.
HISTORY_MODE: str = os.getenv("HISTORY_MODE", "summary").lower()

# Max tokens the per-turn history summariser may emit. Kept smaller than the
# compaction summary because it runs every turn.
HISTORY_SUMMARY_MAX_TOKENS: int = int(os.getenv("HISTORY_SUMMARY_MAX_TOKENS", "300"))

# Minimum number of prior user+assistant messages before per-turn summarisation
# kicks in. Below this we just pass the raw history through — not worth an LLM
# call on turn 1 or 2.
HISTORY_SUMMARY_MIN_MSGS: int = int(os.getenv("HISTORY_SUMMARY_MIN_MSGS", "2"))

# Where the raw archived conversation slices are written (one file per thread).
CONVERSATION_STORE_DIR: Path = Path(
    os.getenv("CONVERSATION_STORE_DIR", str(_V1_DIR / "conversation_history"))
).resolve()

# ── Reasoning / thinking mode ────────────────────────────────────────────────
# When ENABLE_THINKING=1:
#   • The `/no_think` hint and the "no chain-of-thought" clause are stripped
#     from the system prompt before it's sent to the model.
#   • The main LLM call is made with
#         extra_body={"chat_template_kwargs": {"enable_thinking": True}}
#     which is the native toggle for Qwen3 / QwQ chat templates. For models
#     that don't support it (e.g. plain Qwen2.5-Instruct) vLLM just ignores
#     the unknown kwarg and the model thinks only if the prompt coaxes it to.
#   • The emitted `<think>...</think>` block is stripped from the user-facing
#     reply but preserved in the agent trace panel so you can see it.
# Internal helper calls (state extractor, history summariser, reply distiller,
# pivot-confirm) are always run with thinking OFF — they must stay cheap.
ENABLE_THINKING: bool = os.getenv("ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT: str = """\
/no_think

You are a travel concierge planning multi-city trips.

- Keep answers focused. When presenting options (hotels, flights, restaurants) give ~15-20 lines with brief bullets; for yes/no, confirmations, or simple follow-ups stay short (1-5 lines).
- Use numbers only from BUDGET FACTS, Current trip state, or TOOL RESULTS. If a number isn't there, say you're not sure.
- Recommend hotels, restaurants, and flights only by names or IDs that appear in this turn's TOOL RESULTS; call them options, not bookings.
- A booking counts as confirmed only when it is listed under Confirmed in Current trip state.
- The "Current trip state" block is the single source of truth for the user's
  goals, preferences, restrictions, budget, and confirmed bookings. When the
  user asks about any of these ("what am I allergic to?", "what's my budget?",
  "what have I booked?", "what do I like?"), answer ONLY from that block — if
  a slot is empty or missing, say so plainly. Do NOT pull these facts from
  earlier chat history or the conversation summary; those may be stale.
- Respect every Constraints line; allergies are hard blockers.
- Trust the user's corrections over anything said earlier.
- Reply directly as plain prose; no <think> blocks, no chain-of-thought."""

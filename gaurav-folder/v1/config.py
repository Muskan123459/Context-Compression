"""
v1 — shared configuration.

All tuneable values live here; everything else imports from this module.
"""
from __future__ import annotations

import os

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL: str = os.getenv("MODEL", "HuggingFaceTB/SmolLM3-3B")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

# Model's hard context-window cap — anything past this is "overflow"
CONTEXT_LIMIT: int = int(os.getenv("CONTEXT_LIMIT", "8192"))

# Warn in the token counter once we exceed this fraction of CONTEXT_LIMIT
WARN_THRESHOLD: float = float(os.getenv("WARN_THRESHOLD", "0.75"))

SYSTEM_PROMPT: str = """\
You are a travel concierge. You help users plan multi-city trips.

RULES:
- Track the user's stated budget carefully. Mention the remaining budget whenever recommending hotels, flights, or activities.
- When a user says they booked something at a stated price, remember that cost and deduct it from the budget.
- When asked about hotels, flights, or places, ALWAYS call the relevant tool first. Do not answer from memory.
- After getting tool results, write a short, helpful reply referencing the actual data.
- Flag any recommendation that would push the user over their stated total budget.
- Keep replies concise and practical."""

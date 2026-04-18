"""
ccm/summarizer.py — Summary Agent (CCM Layer 1)

After every conversation turn this agent reads the raw history and produces a
structured memory block that gets re-injected into the chat agent's system prompt.

Key design decisions
--------------------
- The summarizer only reads REAL user/assistant turns (not TOOL_RESULT injections)
  so the transcript stays small and clean.
- The prompt is intentionally simple — SmolLM2 at 1.7B follows short-form
  instructions better than rigid multi-field schemas.
- Parsing is fault-tolerant: accepts both "Field: value" and prose forms.

Public interface:
    Summarizer(model, tokenizer, debug=False, max_summary_tokens=48)
    summarizer.update(history) -> SummaryBlock
    summarizer.inject(system_prompt, block) -> str
"""

import re
import textwrap
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SummaryBlock:
    destinations: list[str] = field(default_factory=list)
    preferences:  list[str] = field(default_factory=list)
    decisions:    list[str] = field(default_factory=list)
    budget_state: str       = ""
    constraints:  list[str] = field(default_factory=list)
    next_steps:   list[str] = field(default_factory=list)
    raw_summary:  str       = ""
    turn_count:   int       = 0
    tokens_in:    int       = 0
    tokens_out:   int       = 0

    def is_empty(self) -> bool:
        return not any([
            self.destinations, self.preferences, self.decisions,
            self.budget_state, self.constraints, self.raw_summary,
        ])

    def to_prompt_block(self) -> str:
        lines = ["## Conversation Memory (compressed — always respect this)"]

        if self.destinations:
            lines.append(f"Destinations: {', '.join(self.destinations)}")
        if self.preferences:
            lines.append("User preferences:")
            for p in self.preferences:
                lines.append(f"  • {p}")
        if self.decisions:
            lines.append("Confirmed decisions:")
            for d in self.decisions:
                lines.append(f"  ✓ {d}")
        if self.budget_state:
            lines.append(f"Budget: {self.budget_state}")
        if self.constraints:
            lines.append("HARD CONSTRAINTS — never ignore:")
            for c in self.constraints:
                lines.append(f"  ⚠ {c}")
        if self.next_steps:
            lines.append("Still to do:")
            for s in self.next_steps:
                lines.append(f"  → {s}")

        lines.append(f"(after turn {self.turn_count})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = textwrap.dedent("""\
    Extract ONLY facts that were explicitly stated in the conversation below.
    Do NOT invent, guess, or copy from these instructions.
    If a field has no evidence in the conversation, leave it out entirely.

    Output format — use only the lines that apply:
    DESTINATIONS: <cities mentioned>
    BUDGET: <budget figure if stated>
    PREFERENCES: <preferences if stated>
    CONSTRAINTS: <hard restrictions if stated>
    DECISIONS: <confirmed bookings if stated>
    NEXT_STEPS: <what user still needs>

    Only include a line if the conversation explicitly contains that information.
    No prose. No extra text. No guessing.
""")


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class Summarizer:
    def __init__(
        self,
        model,
        tokenizer,
        debug: bool = False,
        max_summary_tokens: int = 48,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.debug     = debug
        # Short structured lines only — smaller decode budget is much faster on CPU.
        self.max_summary_tokens = max(16, min(int(max_summary_tokens), 256))
        self._current  = SummaryBlock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, history: list[dict]) -> SummaryBlock:
        """Run summarizer on real conversation turns; return updated SummaryBlock."""
        # Only keep genuine user/assistant turns — skip TOOL_RESULT injections
        turns = [
            m for m in history
            if m["role"] in ("user", "assistant")
            and not m["content"].startswith("TOOL_RESULT[")
            and not m["content"].startswith("Now answer")
            and not m["content"].startswith("You already called")
            and not m["content"].startswith("Give your final")
            and not m["content"].startswith("Please give")
        ]
        # Drop system messages
        turns = [m for m in turns if m["role"] != "system"]

        if not turns:
            return self._current

        transcript = self._build_transcript(turns)
        tokens_in  = len(self.tokenizer.encode(transcript))

        raw        = self._generate(transcript)
        tokens_out = len(self.tokenizer.encode(raw))

        if self.debug:
            print(f"\n[SUMMARIZER RAW OUTPUT]:\n{raw}\n")

        block             = self._parse(raw)
        block.turn_count  = sum(1 for m in turns if m["role"] == "user")
        block.tokens_in   = tokens_in
        block.tokens_out  = tokens_out
        block.raw_summary = raw
        self._current     = block
        return block

    def inject(self, base_system_prompt: str, block: SummaryBlock) -> str:
        if block.is_empty():
            return base_system_prompt
        return f"{base_system_prompt}\n\n{block.to_prompt_block()}"

    @property
    def current(self) -> SummaryBlock:
        return self._current

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_transcript(self, turns: list[dict]) -> str:
        """Build a clean short transcript — only real dialogue."""
        lines = []
        for m in turns:
            role    = "User" if m["role"] == "user" else "Assistant"
            cap     = 280 if m["role"] == "user" else 220
            content = m["content"][:cap]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generate(self, transcript: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Conversation:\n{transcript}\n\nExtract now:"},
        ]
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        in_len  = inputs["input_ids"].shape[-1]

        import torch
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out_ids[0][in_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _parse(text: str) -> SummaryBlock:
        block = SummaryBlock()

        def _get(label: str) -> str:
            """Extract the value after 'LABEL:' on the same line."""
            m = re.search(
                rf"^{label}\s*[:\-]\s*(.+)$",
                text, re.IGNORECASE | re.MULTILINE
            )
            return m.group(1).strip() if m else ""

        def _list(label: str) -> list[str]:
            raw = _get(label)
            if not raw:
                return []
            # Split on commas or semicolons
            items = re.split(r"[,;]", raw)
            return [i.strip().lstrip("•→✓⚠-* ") for i in items if i.strip()]

        # Destinations
        dest = _get("DESTINATIONS")
        if dest:
            block.destinations = [d.strip() for d in re.split(r"[,;]", dest) if d.strip()]

        # Budget — keep as single string
        block.budget_state = _get("BUDGET")

        # Preferences
        block.preferences = _list("PREFERENCES")

        # Constraints — critical: allergy, hard limits
        block.constraints = _list("CONSTRAINTS")

        # Decisions
        block.decisions = _list("DECISIONS")

        # Next steps
        block.next_steps = _list("NEXT_STEPS")

        return block

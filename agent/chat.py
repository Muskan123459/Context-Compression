"""
chat.py — SmolLM2 CLI agent with tool calling (ReAct loop) + CCM summarizer.

After every turn the Summary Agent compresses the conversation into a structured
memory block that gets re-injected into the system prompt for the next turn.
The updated prompt is shown on the CLI after each turn.

Usage:
    python chat.py
    python chat.py --model HuggingFaceTB/SmolLM2-360M-Instruct
    python chat.py --system "You are a travel concierge."
    python chat.py --max-new-tokens 512
    python chat.py --no-summary          # disable CCM summarizer
"""

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccm.summarizer import Summarizer, SummaryBlock
from tools import budget_tracker
from tools.registry import execute, tool_schema_block


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    GREEN   = "\033[32m"
    MAGENTA = "\033[35m"
    RED     = "\033[31m"
    BLUE    = "\033[34m"
    DIM     = "\033[2m"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert travel concierge with deep knowledge of destinations, "
    "logistics, budgeting, and trip planning. "
    "Your primary mode is to answer from your own expertise — you know a great deal "
    "about travel and should share it confidently. "
    "Only reach for a tool when you need a live number or a real-time listing "
    "that you genuinely cannot provide from your own knowledge. "
    "Be concise, practical, and opinionated — give recommendations, not just lists.\n\n"
    "IMPORTANT RULES:\n"
    "1. If the user mentions a budget amount, immediately call budget_tracker with "
    "action='set_total' before doing anything else.\n"
    "2. Before recommending ANY food, restaurant, or market, check the Conversation Memory "
    "for dietary constraints (e.g. shellfish allergy). If a constraint exists, "
    "explicitly warn about incompatible options and only recommend safe ones.\n"
    "3. If Conversation Memory shows a constraint like 'shellfish allergy', "
    "NEVER recommend sushi bars, seafood markets, or fish restaurants without a clear warning."
)


def load_model(model_id: str):
    print(f"{C.CYAN}Loading {model_id} …{C.RESET}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"{C.GREEN}Model ready on {device}.{C.RESET}\n", flush=True)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, history: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )


def token_count(tokenizer, history: list[dict]) -> int:
    return len(tokenizer.encode(build_prompt(tokenizer, history)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def generate_raw(tokenizer, model, history: list[dict],
                 max_new_tokens: int = 512) -> str:
    prompt    = build_prompt(tokenizer, history)
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI display helpers — tool calls
# ---------------------------------------------------------------------------

_TOOL_ICONS = {
    "web_search":     "🔍",
    "places_search":  "📍",
    "weather_fetch":  "🌤 ",
    "budget_tracker": "💰",
}

_RESULT_PREVIEW_KEYS = {
    "weather_fetch":  ["temperature_c", "description", "humidity_pct", "source"],
    "budget_tracker": ["remaining", "percent_used", "total_spent", "warnings", "status"],
}


def _print_tool_call(name: str, arguments: dict, round_idx: int) -> None:
    icon     = _TOOL_ICONS.get(name, "⚙ ")
    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in arguments.items())
    print(f"\n{C.YELLOW}{C.BOLD}  {icon}  [{round_idx}] TOOL  →  {name}({args_str}){C.RESET}")


def _print_tool_result(name: str, result_json: str, elapsed_ms: int) -> None:
    try:
        obj = json.loads(result_json)
    except Exception:
        print(f"{C.DIM}  ↳  raw: {result_json[:200]}{C.RESET}\n")
        return

    source = obj.get("source", "")

    for w in obj.get("warnings", []):
        print(f"  {C.RED}{C.BOLD}  ⚠  {w}{C.RESET}")

    if obj.get("guidance"):
        print(f"{C.DIM}  ↳  directive ({elapsed_ms} ms) [{source}]:{C.RESET}")
        for line in obj["guidance"].splitlines():
            print(f"{C.DIM}       {line}{C.RESET}")
    elif name == "budget_tracker" and obj.get("status") in ("recorded", None):
        # Show a clear budget box for any add/deduct operation
        total     = obj.get("budget_total", 0)
        amount    = obj.get("amount_added", 0)
        spent     = obj.get("total_spent", 0)
        remaining = obj.get("remaining", 0)
        pct       = obj.get("percent_used", 0)
        category  = obj.get("category", "")
        print(f"\n{C.MAGENTA}{C.BOLD}  💰  BUDGET UPDATE  ({category}){C.RESET}")
        print(f"{C.MAGENTA}  ┌─────────────────────────────────────────┐{C.RESET}")
        print(f"{C.MAGENTA}  │  Expense    : -${amount:>10,.0f}                │{C.RESET}")
        print(f"{C.MAGENTA}  │  Total      :  ${total:>10,.0f}                │{C.RESET}")
        print(f"{C.MAGENTA}  │  Spent      :  ${spent:>10,.0f}                │{C.RESET}")
        print(f"{C.MAGENTA}  │  Remaining  :  ${remaining:>10,.0f}  ({pct}% used)   │{C.RESET}")
        print(f"{C.MAGENTA}  └─────────────────────────────────────────┘{C.RESET}")
    else:
        preview_keys = _RESULT_PREVIEW_KEYS.get(name)
        preview      = {k: obj[k] for k in preview_keys if k in obj} if preview_keys else obj
        print(f"{C.DIM}  ↳  data ({elapsed_ms} ms) [{source}]:{C.RESET}")
        for line in json.dumps(preview, indent=2).splitlines():
            print(f"{C.DIM}       {line}{C.RESET}")
    print()


# ---------------------------------------------------------------------------
# CLI display helpers — CCM summary
# ---------------------------------------------------------------------------

def _print_summary_update(block: SummaryBlock, new_system: str) -> None:
    """Print the CCM summary block and the updated system prompt."""
    w = 62
    print(f"\n{C.BLUE}{C.BOLD}{'─' * w}")
    print(f"  🧠  CCM SUMMARY  (turn {block.turn_count} | "
          f"{block.tokens_in}→{block.tokens_out} tokens)")
    print(f"{'─' * w}{C.RESET}")

    if block.destinations:
        print(f"{C.BLUE}  Destinations : {', '.join(block.destinations)}{C.RESET}")
    for p in block.preferences:
        print(f"{C.BLUE}  Preference   : {p}{C.RESET}")
    for d in block.decisions:
        print(f"{C.BLUE}  Decision  ✓  : {d}{C.RESET}")
    if block.budget_state:
        print(f"{C.BLUE}  Budget       : {block.budget_state}{C.RESET}")
    for c in block.constraints:
        print(f"{C.RED}{C.BOLD}  Constraint ⚠ : {c}{C.RESET}")
    for s in block.next_steps:
        print(f"{C.BLUE}  Next step  → : {s}{C.RESET}")

    # Show the updated system prompt that will be used next turn
    print(f"\n{C.MAGENTA}{C.BOLD}  📋  UPDATED SYSTEM PROMPT (next turn):{C.RESET}")
    for line in new_system.splitlines():
        print(f"{C.MAGENTA}  {line}{C.RESET}")

    print(f"{C.BLUE}{C.BOLD}{'─' * w}{C.RESET}\n")


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

_PROMPT_TOOLS = {"web_search", "places_search"}   # guidance-only, no looping

# ---------------------------------------------------------------------------
# Auto budget sync — detects confirmed spends in assistant responses
# ---------------------------------------------------------------------------

# Words that signal a booking was confirmed in this turn
_CONFIRMATION_WORDS = (
    "booked", "confirmed", "reserved", "booking confirmed",
    "have booked", "successfully booked", "total cost", "total: $",
)

# Patterns to extract a dollar amount — tried in priority order
_SPEND_PATTERNS = [
    # "total cost is $2,700" / "total cost: $2700"
    re.compile(r"total\s+cost[^\$]*\$\s?([\d,]+)", re.IGNORECASE),
    # "total: $2,700" / "Total $2700"
    re.compile(r"\btotal[:\s]+\$\s?([\d,]+)", re.IGNORECASE),
    # "cost is $2,700"
    re.compile(r"cost(?:s)?\s+(?:is|of|:)\s*\$\s?([\d,]+)", re.IGNORECASE),
    # generic "$2,700" (last resort — only if confirmation words present)
    re.compile(r"\$\s?([\d,]+)"),
]


def _detect_confirmed_spend(response: str) -> tuple[float, str] | None:
    """
    Scan an assistant response for a confirmed booking with a dollar amount.
    Returns (amount, label) or None.
    Requires at least one confirmation word to be present.
    """
    lower = response.lower()
    if not any(w in lower for w in _CONFIRMATION_WORDS):
        return None

    for pattern in _SPEND_PATTERNS:
        m = pattern.search(response)
        if m:
            try:
                amount = float(m.group(1).replace(",", ""))
                if amount > 0:
                    # Pull a short description from context around the match
                    start = max(0, m.start() - 40)
                    snippet = response[start : m.end()].strip()
                    return amount, snippet
            except ValueError:
                continue
    return None


def agent_turn(tokenizer, model, history: list[dict],
               max_new_tokens: int, max_tool_rounds: int = 6) -> tuple[str, bool]:
    """Returns (response_text, budget_was_recorded_by_model)."""
    seen_calls: set[tuple] = set()   # dedup safety net
    budget_recorded = False

    for round_idx in range(1, max_tool_rounds + 1):
        raw   = generate_raw(tokenizer, model, history, max_new_tokens)
        match = TOOL_CALL_RE.search(raw)

        if not match:
            return raw, budget_recorded   # clean final answer

        pre_text = raw[:match.start()].strip()
        if pre_text:
            print(f"{C.DIM}  (thinking) {pre_text}{C.RESET}")

        # Parse — surface raw text if malformed
        try:
            call   = json.loads(match.group(1).strip())
            t_name = call["name"]
            t_args = call.get("arguments", {})
        except (json.JSONDecodeError, KeyError):
            return raw, budget_recorded

        # Deduplicate: same tool + same args twice → stop looping
        call_key = (t_name, json.dumps(t_args, sort_keys=True))
        if call_key in seen_calls:
            history.append({
                "role":    "user",
                "content": "You already called that tool. Give your final answer now.",
            })
            return generate_raw(tokenizer, model, history, max_new_tokens), budget_recorded
        seen_calls.add(call_key)

        _print_tool_call(t_name, t_args, round_idx)

        t0          = time.monotonic()
        result_json = execute(t_name, t_args)
        elapsed_ms  = int((time.monotonic() - t0) * 1000)

        if t_name == "budget_tracker" and t_args.get("action") in ("add", "deduct"):
            budget_recorded = True

        _print_tool_result(t_name, result_json, elapsed_ms)

        try:
            result_obj = json.loads(result_json)
            injection  = result_obj.get("guidance") or result_json
        except Exception:
            injection  = result_json

        history.append({"role": "assistant", "content": raw})
        history.append({
            "role":    "user",
            "content": f"TOOL_RESULT[{t_name}]: {injection}",
        })

        # Prompt tools (web_search, places_search) give guidance, not data.
        # One call is enough — force the final answer immediately.
        if t_name in _PROMPT_TOOLS:
            history.append({
                "role":    "user",
                "content": "Now answer using the guidance above. Do NOT call any more tools.",
            })
            return generate_raw(tokenizer, model, history, max_new_tokens), budget_recorded

    history.append({
        "role":    "user",
        "content": "Give your final answer now based on everything gathered.",
    })
    return generate_raw(tokenizer, model, history, max_new_tokens), budget_recorded


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

BANNER = f"""
{C.CYAN}{C.BOLD}{'=' * 62}
  SmolLM2 Agent  +  CCM Summarizer
  tools: web_search · places_search · weather_fetch · budget_tracker
{'=' * 62}{C.RESET}
  {C.BOLD}/reset{C.RESET}    clear history, budget & summary
  {C.BOLD}/tokens{C.RESET}   show context token count
  {C.BOLD}/budget{C.RESET}   show budget status
  {C.BOLD}/summary{C.RESET}  show current CCM memory block
  {C.BOLD}/system{C.RESET}   show current active system prompt
  {C.BOLD}quit{C.RESET}      exit
"""


def chat_loop(
    tokenizer, model,
    max_new_tokens: int,
    system_prompt:  str,
    use_summary:    bool = True,
    debug_summary:  bool = False,
) -> None:
    summarizer   = Summarizer(model, tokenizer, debug=debug_summary) if use_summary else None
    tool_block   = tool_schema_block()

    # The "active" system prompt starts bare and gains the summary block each turn
    base_system  = system_prompt
    active_system = f"{base_system}\n\n{tool_block}"
    history: list[dict] = [{"role": "system", "content": active_system}]

    print(BANNER)
    print(f"{C.DIM}System: {system_prompt}{C.RESET}")
    if use_summary:
        print(f"{C.BLUE}CCM Summarizer: ON{C.RESET}")
    else:
        print(f"{C.DIM}CCM Summarizer: OFF (--no-summary){C.RESET}")
    print()

    while True:
        try:
            user_input = input(f"{C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        if user_input == "/reset":
            if summarizer:
                summarizer._current = SummaryBlock()   # fresh empty block
            budget_tracker.BUDGET.reset()
            active_system = f"{base_system}\n\n{tool_block}"
            history = [{"role": "system", "content": active_system}]
            print(f"{C.YELLOW}[History, budget, and summary cleared]{C.RESET}\n")
            continue

        if user_input == "/tokens":
            print(f"{C.DIM}[Context tokens: {token_count(tokenizer, history)}]{C.RESET}\n")
            continue

        if user_input == "/budget":
            status = budget_tracker.BUDGET.get_status()
            print(f"{C.MAGENTA}{json.dumps(status, indent=2)}{C.RESET}\n")
            continue

        if user_input == "/summary":
            if summarizer and not summarizer.current.is_empty():
                print(f"{C.BLUE}{summarizer.current.to_prompt_block()}{C.RESET}\n")
            else:
                print(f"{C.DIM}[No summary yet]{C.RESET}\n")
            continue

        if user_input == "/system":
            print(f"{C.MAGENTA}{active_system}{C.RESET}\n")
            continue

        # ── normal turn ────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        print(f"\n{C.GREEN}{C.BOLD}Assistant:{C.RESET} ", end="", flush=True)
        response, budget_recorded = agent_turn(tokenizer, model, history, max_new_tokens)
        print(response)
        print()

        # ── Auto budget sync ────────────────────────────────────────────
        if not budget_recorded:
            spend = _detect_confirmed_spend(response)
            if spend:
                amount, snippet = spend
                category = "booking"
                # Guess category from snippet
                if any(w in snippet.lower() for w in ["hotel", "stay", "night"]):
                    category = "hotel"
                elif any(w in snippet.lower() for w in ["flight", "ticket", "airfare"]):
                    category = "flights"
                elif any(w in snippet.lower() for w in ["tour", "activity", "ticket"]):
                    category = "activities"

                before = budget_tracker.BUDGET.total - budget_tracker.BUDGET.spent
                result = budget_tracker.BUDGET.run(
                    "add", amount=amount, category=category,
                    description=f"auto-detected: {snippet[:60]}"
                )
                status = budget_tracker.BUDGET.get_status()
                total     = status["budget_total"]
                spent     = status["total_spent"]
                remaining = status["remaining"]
                pct       = status["percent_used"]

                print(f"\n{C.MAGENTA}{C.BOLD}  💰  BUDGET UPDATE  ({category}){C.RESET}")
                print(f"{C.MAGENTA}  ┌─────────────────────────────────────────┐{C.RESET}")
                print(f"{C.MAGENTA}  │  Expense    : -${amount:>10,.0f}                │{C.RESET}")
                print(f"{C.MAGENTA}  │  Total      :  ${total:>10,.0f}                │{C.RESET}")
                print(f"{C.MAGENTA}  │  Spent      :  ${spent:>10,.0f}                │{C.RESET}")
                print(f"{C.MAGENTA}  │  Remaining  :  ${remaining:>10,.0f}  ({pct}% used)   │{C.RESET}")
                print(f"{C.MAGENTA}  └─────────────────────────────────────────┘{C.RESET}")
                for w in result.get("warnings", []):
                    print(f"  {C.RED}{C.BOLD}  ⚠  {w}{C.RESET}")
                print()

        history.append({"role": "assistant", "content": response})

        # ── CCM: run summary agent, rebuild system prompt ───────────────
        if summarizer:
            print(f"{C.DIM}  [CCM] running summarizer…{C.RESET}", flush=True)
            t0    = time.monotonic()
            block = summarizer.update(history)
            ms    = int((time.monotonic() - t0) * 1000)

            # Rebuild active system prompt with fresh memory block
            new_system    = summarizer.inject(base_system, block)
            active_system = f"{new_system}\n\n{tool_block}"

            # Replace the system message in history for the next turn
            history[0] = {"role": "system", "content": active_system}

            _print_summary_update(block, new_system)
            print(f"{C.DIM}  [CCM] done in {ms} ms | "
                  f"context now: {token_count(tokenizer, history)} tokens{C.RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SmolLM2 agent + CCM summarizer")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--system", default=None,
                   help="System prompt (asked interactively if omitted)")
    p.add_argument("--no-summary", action="store_true",
                   help="Disable the CCM summarizer (baseline mode)")
    p.add_argument("--debug-summary", action="store_true",
                   help="Print raw summarizer model output for debugging")
    return p.parse_args()


def resolve_system_prompt(arg: str | None) -> str:
    if arg:
        return arg.strip()
    print("Enter a system prompt (or press Enter for default):")
    print(f"  {C.DIM}Default: {DEFAULT_SYSTEM_PROMPT}{C.RESET}")
    val = input("> ").strip()
    return val if val else DEFAULT_SYSTEM_PROMPT


if __name__ == "__main__":
    args          = parse_args()
    system_prompt = resolve_system_prompt(args.system)
    tokenizer, model = load_model(args.model)
    chat_loop(
        tokenizer, model,
        args.max_new_tokens,
        system_prompt,
        use_summary=not args.no_summary,
        debug_summary=args.debug_summary,
    )

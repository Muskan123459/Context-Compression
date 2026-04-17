#!/usr/bin/env python3
"""
Test C — Budget Anchor (Short)
================================
A condensed 8-turn version of Test B designed to:

  1. Stay under the 8 192-token context limit (no overflow crash)
  2. Still bury the budget state under enough tool-response bloat
     that a compression-free baseline forgets the cumulative spend

Context budget maths
--------------------
  2 tool calls before the critical check × ~2 500 tokens each  ≈ 5 000 tokens
  + system prompt + conversation text                           ≈ 1 200 tokens
  ─────────────────────────────────────────────────────────────── ≈ 6 200 tokens
  Well under the 8 192 limit, so the test completes every turn.

  The budget numbers are intentionally scattered across turns
  (some stated, some implied, one block at turn 6) so the model
  must piece together: $1 850 + $700 + $550 + $320 = $3 420 spent,
  leaving only $380 of the stated $3 800.

Conversation summary
--------------------
Turn 1 : State budget $3 800, outline trip (no tool)
Turn 2 : Search flights Delhi → multi-city   ← tool call #1  (~2 500 tok)
Turn 3 : Confirm Qatar Airways booking at $1 850 (no tool)
Turn 4 : Search Tokyo hotels, Shinjuku area  ← tool call #2  (~2 500 tok)
Turn 5 : Confirm Shinjuku hotel at $700 (no tool)
Turn 6 : Bulk-confirm Paris $550 + Amsterdam $320 (no tool)
Turn 7 : Bali intro — want Ubud, relaxed, 3 nights (no tool)
Turn 8 : ★ CRITICAL ★  "Find me a hotel in Bali."

Expected state at Turn 8
-------------------------
  Spent  : $1 850 + $700 + $550 + $320 = $3 420
  Budget : $3 800
  Left   : $380

Pass criteria
-------------
Agent MUST show budget awareness at Turn 8:
  • mention ~$380 remaining, OR
  • warn that available options exceed the budget, OR
  • recommend only within-budget options.

Hard fail
---------
Agent recommends Four Seasons (~$900/night) or Katamama (~$450/night)
without any budget caveat — proof the baseline lost the spend state.

Usage
-----
  python eval/test_c.py              # live vLLM
  python eval/test_c.py --dry-run    # offline / CI  (simulates baseline failure)
  python eval/test_c.py --verbose    # show per-turn response previews
  python eval/test_c.py --log out/c.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_V1   = _HERE.parent
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

# ---------------------------------------------------------------------------
# Scripted conversation — 8 turns
# ---------------------------------------------------------------------------
SCRIPT: list[tuple[int, str]] = [
    (1, (
        "I'm planning a solo multi-city trip: Paris (3 nights), Amsterdam (2 nights), "
        "Tokyo (4 nights), Bali (3 nights).\n\n"
        "My total budget is $3,800 including flights and hotels.\n\n"
        "Please keep a running tally of what I spend as we go."
    )),
    (2, (
        "Find me flights from Delhi covering the full route: "
        "Delhi → Paris → Amsterdam → Tokyo → Bali → Delhi. "
        "Fewer layovers preferred."
    )),
    (3, (
        "That Qatar Airways multi-city option at $1,850 total looks good. "
        "Let's assume I book that. "
        "Please track $1,850 against my $3,800 budget."
    )),
    (4, (
        "Now Tokyo — I'll be there 4 nights. "
        "Find hotels near Shinjuku with easy train access."
    )),
    (5, (
        "The Shinjuku Granbell works. "
        "Assume $700 total for my 4-night Tokyo stay. "
        "Add that to the budget tally."
    )),
    (6, (
        "I've sorted the other two cities separately:\n"
        "  Paris (3 nights) — boutique near Saint-Germain: $550 total\n"
        "  Amsterdam (2 nights) — canal-area hotel: $320 total\n\n"
        "Please add both to my running total."
    )),
    (7, (
        "Great. Last leg is Bali — I want to end the trip relaxed. "
        "Thinking Ubud for 3 nights. "
        "Somewhere with good views, private room, ideally breakfast included."
    )),
    (8, (
        "Find me a hotel in Bali."
    )),
]

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
TOTAL_BUDGET   = 3800.0
EXPECTED_SPENT = 1850.0 + 700.0 + 550.0 + 320.0   # = 3 420
EXPECTED_LEFT  = TOTAL_BUDGET - EXPECTED_SPENT      # = 380

# ---------------------------------------------------------------------------
# Dry-run canned responses
# Turn 8 is intentionally forgetful to simulate baseline failure
# ---------------------------------------------------------------------------
_DRY_RUN_RESPONSES: dict[int, str] = {
    1: "Got it! Paris → Amsterdam → Tokyo → Bali, $3,800 total. I'll track your spending.",
    2: "Here are multi-city flight options from Delhi for your route.",
    3: "Booked Qatar Airways at $1,850. Running total: $1,850 / $3,800.",
    4: "Here are hotel options near Shinjuku, Tokyo.",
    5: "Shinjuku Granbell confirmed at $700. Running total: $2,550 / $3,800.",
    6: "Added Paris $550 and Amsterdam $320. Running total: $3,420 / $3,800. You have $380 left.",
    7: "Ubud is perfect for a relaxed ending — rice terraces, jungle vibe, great breakfast spots.",
    # Baseline failure: agent ignores the $380 remaining and recommends freely.
    # No mention of remaining budget, no caveat — pure forgetfulness.
    8: (
        "Here are great hotel options in Bali (Ubud area):\n\n"
        "1. **Four Seasons Sayan** — $900/night. Iconic jungle-canopy infinity pool, "
        "world-class spa, impeccable service. Highly recommended for a luxurious stay.\n\n"
        "2. **Katamama** — $450/night. Boutique luxury in Seminyak with beautiful "
        "Balinese craftsmanship throughout. A real gem.\n\n"
        "3. **Bisma Eight** — $220/night. Rice terrace views, boutique feel, "
        "good value for Ubud."
    ),
}

# ---------------------------------------------------------------------------
# Assertion helpers (shared logic with test_b)
# ---------------------------------------------------------------------------

def _mentions_remaining_budget(response: str) -> bool:
    r = response.lower()
    patterns = [
        r"remain",
        r"\$\s*38[0-9]",
        r"\b380\b",
        r"budget.*tight",
        r"tight.*budget",
        r"over.*budget",
        r"exceed.*budget",
        r"budget.*exceed",
        r"afford",
        r"limited.*budget",
        r"budget.*concern",
        r"can'?t.*afford",
        r"only.*\$\s*3[0-9]{2}",
        r"3[78][0-9].*left",
        r"left.*3[78][0-9]",
        r"short",
    ]
    return any(re.search(p, r) for p in patterns)


def _recommends_luxury_without_caveat(response: str) -> bool:
    r = response.lower()
    has_luxury = "four seasons" in r or "katamama" in r
    has_caveat = _mentions_remaining_budget(response)
    return has_luxury and not has_caveat


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn:     int
    message:  str
    response: str
    tokens:   int = 0
    latency:  float = 0.0
    trace:    list[str] = field(default_factory=list)  # full agent trace incl. tool calls


@dataclass
class RunResult:
    turns:          list[TurnResult] = field(default_factory=list)
    peak_tokens:    int = 0
    final_response: str = ""
    overflow_turn:  Optional[int] = None


@dataclass
class Assertion:
    name:   str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _print_trace(trace: list[str]) -> None:
    """Print tool-call lines from the agent trace (filtered for readability)."""
    tool_lines = [l for l in trace if l.startswith("tool call →") or l.startswith("tool result")]
    if not tool_lines:
        print("      [no tools routed this turn]")
        return
    for line in tool_lines:
        short = line if len(line) <= 180 else line[:180] + "…"
        print(f"      {short}")


def run_conversation(
    *, dry_run: bool = False, verbose: bool = False, show_trace: bool = False
) -> RunResult:
    history: list[dict] = []
    result = RunResult()

    if not dry_run:
        from agent import run_agent  # noqa: PLC0415

    for turn_no, user_msg in SCRIPT:
        t0 = time.monotonic()

        augmented_msg = user_msg   # will be overwritten with tool-blob version if tools fire

        if dry_run:
            reply_text  = _DRY_RUN_RESPONSES.get(turn_no, "(no canned response)")
            token_count = 0
            trace: list[str] = []
        else:
            try:
                agent_reply   = run_agent(user_msg, history, verbose=False)
                reply_text    = agent_reply.text
                token_count   = agent_reply.usage.total
                trace         = agent_reply.trace
                augmented_msg = agent_reply.augmented_message or user_msg
            except Exception as exc:
                overflow_msg = (
                    f"[CONTEXT OVERFLOW at Turn {turn_no}] {type(exc).__name__}: {exc}"
                )
                print(f"\n  *** {overflow_msg}\n", flush=True)
                tr = TurnResult(
                    turn=turn_no, message=user_msg,
                    response=overflow_msg,
                    tokens=result.peak_tokens,
                    latency=time.monotonic() - t0,
                )
                result.turns.append(tr)
                result.final_response = overflow_msg
                result.overflow_turn  = turn_no
                return result

        latency = time.monotonic() - t0
        tr = TurnResult(
            turn=turn_no, message=user_msg,
            response=reply_text, tokens=token_count, latency=latency,
            trace=trace,
        )
        result.turns.append(tr)
        result.peak_tokens    = max(result.peak_tokens, token_count)
        result.final_response = reply_text

        # Store the augmented message (with tool blobs) so tool data accumulates in context
        history.append({"role": "user",      "content": augmented_msg})
        history.append({"role": "assistant", "content": reply_text})

        print(f"Turn {turn_no}  {_token_bar_str(token_count)}  {latency:.1f}s", flush=True)
        if show_trace:
            _print_trace(trace)
        if verbose:
            preview = textwrap.shorten(reply_text, width=120, placeholder="…")
            print(f"      ↳ {preview}")

    return result


def _token_bar_str(tokens: int, limit: int = 8192, width: int = 16) -> str:
    pct  = tokens / limit * 100 if limit else 0
    fill = int(pct / 100 * width)
    bar  = "█" * fill + "░" * (width - fill)
    warn = " ⚠" if pct >= 75 else "  "
    return f"[{bar}] {tokens:>5,} tok ({pct:4.1f}%){warn}"


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def evaluate(result: RunResult) -> list[Assertion]:
    assertions: list[Assertion] = []
    r8 = result.final_response
    overflowed = result.overflow_turn is not None

    # A1 — context grew (bloat visible)
    assertions.append(Assertion(
        name="A1 · token-count climbs",
        passed=result.peak_tokens > 2000,
        detail=(
            f"peak={result.peak_tokens:,} tokens "
            f"({'BLOAT VISIBLE' if result.peak_tokens > 2000 else 'unexpectedly low — check fixtures'})"
        ),
    ))

    if overflowed:
        assertions.append(Assertion(
            name="A2 · reached Turn 8 without overflow",
            passed=False,
            detail=(
                f"CONTEXT OVERFLOW at Turn {result.overflow_turn}. "
                "Short script should not overflow — check fixture sizes."
            ),
        ))
        assertions.append(Assertion(
            name="A3 · no uncaveated luxury recommendation",
            passed=False,
            detail="N/A — crashed before Turn 8.",
        ))
        return assertions

    # A2 — budget awareness at Turn 8
    aware = _mentions_remaining_budget(r8)
    assertions.append(Assertion(
        name="A2 · budget awareness at Turn 8",
        passed=aware,
        detail=(
            "Agent flagged remaining budget / constraint"
            if aware else
            f"Agent gave no budget signal. Expected ~${EXPECTED_LEFT:.0f} remaining."
        ),
    ))

    # A3 — no blind luxury recommendation
    blind = _recommends_luxury_without_caveat(r8)
    assertions.append(Assertion(
        name="A3 · no uncaveated luxury recommendation",
        passed=not blind,
        detail=(
            "Agent did not push luxury without budget caveat"
            if not blind else
            "FAIL — agent recommended Four Seasons / Katamama without flagging "
            f"~${EXPECTED_LEFT:.0f} remains. Baseline lost cumulative spend state."
        ),
    ))

    return assertions


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------
RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
YELLOW= "\033[33m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def print_report(result: RunResult, assertions: list[Assertion], *, dry_run: bool) -> bool:
    print()
    print("=" * 68)
    print(f"  TEST C — Budget Anchor (Short)  {'[DRY RUN]' if dry_run else ''}")
    print("=" * 68)

    print()
    print("  Token trajectory")
    print("  ─" * 34)
    for tr in result.turns:
        print(f"  Turn {tr.turn}  {_token_bar_str(tr.tokens)}")

    print()
    if result.overflow_turn is not None:
        print(f"  *** CONTEXT OVERFLOW at Turn {result.overflow_turn} ***")
        print("  ─" * 34)
        print(f"  {_c(result.final_response[:200], RED)}")
    else:
        print("  Turn 8 response (critical check)")
        print("  ─" * 34)
        wrapped = textwrap.fill(result.final_response, width=64,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

    print()
    print("  Expected budget state at Turn 8")
    print("  ─" * 34)
    print(f"  Total budget : ${TOTAL_BUDGET:,.0f}")
    print(f"  Spent        : ${EXPECTED_SPENT:,.0f}  "
          "(flights $1,850 + Tokyo $700 + Paris $550 + Amsterdam $320)")
    print(f"  Remaining    : ${EXPECTED_LEFT:,.0f}  ← agent MUST acknowledge this")

    print()
    print("  Assertions")
    print("  ─" * 34)
    overall_pass = True
    for a in assertions:
        icon  = _c("PASS", GREEN) if a.passed else _c("FAIL", RED)
        label = _c(a.name, BOLD)
        print(f"  {icon}  {label}")
        print(f"       {_c(a.detail, DIM)}")
        if not a.passed:
            overall_pass = False

    print()
    print("  ─" * 34)
    if overall_pass:
        verdict = _c("OVERALL: PASS", GREEN + BOLD)
    else:
        verdict = _c("OVERALL: FAIL", RED + BOLD)
        note = (
            "Expected baseline outcome — no compression means the agent "
            "loses cumulative budget state across tool-response bloat."
        )
        print(f"  {_c(note, YELLOW)}")
    print(f"  {verdict}")
    print("=" * 68)
    print()

    return overall_pass


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------
def _save_log(result: RunResult, assertions: list[Assertion], path: Path) -> None:
    data = {
        "test": "test_c_budget_anchor_short",
        "ground_truth": {
            "total_budget":       TOTAL_BUDGET,
            "expected_spent":     EXPECTED_SPENT,
            "expected_remaining": EXPECTED_LEFT,
        },
        "peak_tokens":    result.peak_tokens,
        "overflow_turn":  result.overflow_turn,
        "turns": [
            {
                "turn":             tr.turn,
                "tokens":           tr.tokens,
                "latency_s":        round(tr.latency, 2),
                "response_preview": tr.response[:200],
                "tool_calls":       [l for l in tr.trace if l.startswith("tool call →")],
            }
            for tr in result.turns
        ],
        "final_response": result.final_response,
        "assertions": [
            {"name": a.name, "passed": a.passed, "detail": a.detail}
            for a in assertions
        ],
        "overall_pass": all(a.passed for a in assertions),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Log saved → {path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test C — Budget Anchor (short) for the travel agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              python eval/test_c.py                   # live vLLM run
              python eval/test_c.py --dry-run         # offline / CI mode
              python eval/test_c.py --verbose         # show per-turn previews
              python eval/test_c.py --log out/c.json  # save JSON report
        """),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use canned responses (offline / CI mode).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print a preview of each turn's response.")
    parser.add_argument("--show-trace", "-t", action="store_true",
                        help="Print tool calls made each turn (name + args + result preview).")
    parser.add_argument("--log", type=Path, default=None, metavar="FILE",
                        help="Write a JSON log to FILE.")
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else f"LIVE (model: {os.getenv('MODEL', 'SmolLM3-3B')})"
    print(f"\n{'='*68}")
    print(f"  Test C — Budget Anchor (Short)  |  {mode}")
    print(f"  {len(SCRIPT)} turns  |  budget=${TOTAL_BUDGET:,.0f}  "
          f"|  expected_remaining=${EXPECTED_LEFT:.0f}")
    print(f"{'='*68}\n")

    result     = run_conversation(dry_run=args.dry_run, verbose=args.verbose,
                                  show_trace=args.show_trace)
    assertions = evaluate(result)
    passed     = print_report(result, assertions, dry_run=args.dry_run)

    if args.log:
        _save_log(result, assertions, args.log)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

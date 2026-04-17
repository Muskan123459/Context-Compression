#!/usr/bin/env python3
"""
Test B — Budget Anchor
======================
Replays a scripted 20-turn multi-city trip conversation and verifies that the
agent correctly tracks a $3 800 total budget across all bookings.

Conversation summary
--------------------
Turn  1  : Set budget $3 800, trip to Paris / Amsterdam / Tokyo / Bali
Turn  2  : Flight search Delhi → multi-city route
Turn  3  : Book Qatar Airways ~$1 850  → running total $1 850
Turn  4  : Paris hotels (3 nights, Latin Quarter / Eiffel Tower area)
Turn  5  : Book Paris boutique hotel $550  → running total $2 400
Turn  6  : Amsterdam hotels (2 nights, canal area / Centraal)
Turn  7  : Book Amsterdam hotel $320  → running total $2 720
Turn  8  : Tokyo hotels (4 nights, Shinjuku / Ginza)
Turn  9  : Book Shinjuku hotel $700  → running total $3 420
Turn 10  : Tokyo ramen + observation deck recommendations
Turn 11  : Bali intro (end of trip, want relaxed vibe)
Turn 12  : Compare Ubud vs Seminyak
Turn 13  : Decide on Ubud, 3 nights
Turn 14  : Suggest hotels with nature vibe in Ubud
Turn 15  : Show one premium villa option too
Turn 16  : Budget check-in — "remind me where my total stands"
Turn 17  : Concern about local transport + food budget
Turn 18  : Preference: transparent pricing, taxes + breakfast included
Turn 19  : "Help me finalise Bali"
Turn 20  : ★ CRITICAL ★  "Find me a hotel in Bali."

Expected state at Turn 20
--------------------------
  Spent  : $1 850 (flights) + $550 (Paris) + $320 (Amsterdam) + $700 (Tokyo)
         = $3 420
  Budget : $3 800
  Left   : $380

Pass criteria
-------------
The agent MUST demonstrate budget awareness in its Turn-20 response:
  • mention the remaining budget (~$380), OR
  • warn that available options exceed the remaining budget, OR
  • explicitly recommend only options within the remaining balance.

Hard fail
---------
Agent recommends a luxury option (e.g. Four Seasons ~$900/night) without
ANY budget caveat.  This means the compression-free baseline lost the
cumulative spend state — the exact failure mode this test exposes.

Usage
-----
  # Against live vLLM (requires model running on localhost:8000):
  python eval/test_b.py

  # Dry-run (skips LLM calls, uses canned responses — for CI / quick checks):
  python eval/test_b.py --dry-run

  # Show full per-turn responses:
  python eval/test_b.py --verbose
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

# ── Make sure we can import from the parent v1/ directory ─────────────────────
_HERE = Path(__file__).resolve().parent          # v1/eval/
_V1   = _HERE.parent                             # v1/
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

# ---------------------------------------------------------------------------
# Scripted conversation
# Each entry: (turn_number, user_message)
# ---------------------------------------------------------------------------
SCRIPT: list[tuple[int, str]] = [
    (1, (
        "I'm planning a long solo trip this summer across Paris, Amsterdam, Tokyo, and Bali.\n\n"
        "Rough plan is:\n"
        "  Paris      → 3 days\n"
        "  Amsterdam  → 2 days\n"
        "  Tokyo      → 4 days\n"
        "  Bali       → 3 days\n\n"
        "My total budget is $3,800 max including flights + hotels, and I want a good balance "
        "between comfort and experience.\n"
        "Not luxury, but I also don't want backpacker hostels.\n\n"
        "Please keep track of my spending as we go."
    )),
    (2, (
        "Let's start with flights.\n\n"
        "Find me a good multi-city route from Delhi covering:\n"
        "  Delhi → Paris → Amsterdam → Tokyo → Bali → Delhi\n\n"
        "I prefer fewer exhausting layovers even if it costs a bit more."
    )),
    (3, (
        "That Qatar Airways option around $1,850 total looks solid.\n\n"
        "Let's assume I book that.\n\n"
        "Please remember that against the total budget."
    )),
    (4, (
        "For Paris, I want to stay near central areas like the Latin Quarter "
        "or close to the Eiffel Tower metro lines.\n\n"
        "Safe, walkable, and easy for sightseeing.\n\n"
        "Find hotel options for 3 nights."
    )),
    (5, (
        "Let's go with that boutique hotel near Saint-Germain.\n\n"
        "Assume $550 total for Paris stay.\n\n"
        "Add that too."
    )),
    (6, (
        "Now Amsterdam.\n\n"
        "I'll stay 2 nights and want somewhere near the canal area "
        "or close to Centraal Station.\n\n"
        "Not too touristy, but convenient."
    )),
    (7, (
        "Perfect.\n\n"
        "Let's assume that hotel costs $320 total.\n\n"
        "Please track that too."
    )),
    (8, (
        "For Tokyo, I'll spend 4 nights.\n\n"
        "Suggest hotels near Shinjuku or Ginza.\n\n"
        "I want easy train access and safe late-night movement."
    )),
    (9, (
        "That Shinjuku hotel looks good.\n\n"
        "Let's assume $700 total for Tokyo.\n\n"
        "Include that in budget."
    )),
    (10, (
        "Also, suggest 2 good ramen places and one nice observation deck in Tokyo.\n\n"
        "Nothing overly tourist-trap."
    )),
    (11, (
        "Great.\n\n"
        "Now for the final part — Bali.\n\n"
        "I want the trip to end more relaxed, so probably Ubud or Seminyak.\n\n"
        "Somewhere peaceful with nice views."
    )),
    (12, (
        "Compare Ubud vs Seminyak for me.\n\n"
        "I care more about calm and scenery than nightlife.\n\n"
        "Also keep the budget in mind."
    )),
    (13, (
        "Ubud sounds better.\n\n"
        "Let's go with that.\n\n"
        "I'll stay there for 3 nights."
    )),
    (14, (
        "Please suggest hotels with a nice nature vibe.\n\n"
        "Private room only, good breakfast preferred, not party-hostel type places."
    )),
    (15, (
        "Maybe show me one premium villa option too, just so I can compare.\n\n"
        "Not necessarily booking it."
    )),
    (16, (
        "Before that, remind me where my total budget currently stands."
    )),
    (17, (
        "I still need room for local transport and food, so I don't want to overspend here.\n\n"
        "That matters."
    )),
    (18, (
        "Hidden fees are annoying, so I'd prefer transparent pricing.\n\n"
        "Taxes and breakfast included if possible."
    )),
    (19, (
        "Okay, based on all that, help me finalise Bali."
    )),
    (20, (
        "Find me a hotel in Bali."
    )),
]

# ---------------------------------------------------------------------------
# Ground-truth budget state (what the agent *should* know by Turn 20)
# ---------------------------------------------------------------------------
TOTAL_BUDGET   = 3800.0
EXPECTED_SPENT = 1850.0 + 550.0 + 320.0 + 700.0   # = 3 420
EXPECTED_LEFT  = TOTAL_BUDGET - EXPECTED_SPENT      # = 380

# Bali hotel prices from the fixture (from v1/tools.py)
# Agent should NOT recommend these at face value without flagging budget:
LUXURY_BALI = {
    "four seasons sayan": 900,   # per night × 3 = $2 700  — way over budget
    "katamama":           450,   # per night × 3 = $1 350  — over budget
}
BUDGET_BALI = {
    "bisma eight":  220,   # per night × 3 = $660 — still over remaining $380
}

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _mentions_remaining_budget(response: str) -> bool:
    """Does the response show awareness of the depleted budget?"""
    r = response.lower()
    patterns = [
        r"remain",
        r"\$\s*38[0-9]",            # $380 ± a few dollars rounding
        r"380",
        r"budget.*tight",
        r"tight.*budget",
        r"over.*budget",
        r"exceed.*budget",
        r"budget.*exceed",
        r"afford",
        r"limited.*budget",
        r"budget.*limited",
        r"budget.*concern",
        r"can'?t.*afford",
        r"short",
        r"only.*\$\s*3[0-9]{2}",   # "only $3xx"
        r"3[78][0-9].*left",
        r"left.*3[78][0-9]",
    ]
    return any(re.search(p, r) for p in patterns)


def _recommends_luxury_without_caveat(response: str) -> bool:
    """True if agent pushes a luxury option with no budget flag whatsoever."""
    r = response.lower()
    has_luxury = "four seasons" in r or "katamama" in r
    has_caveat = _mentions_remaining_budget(response)
    return has_luxury and not has_caveat


# ---------------------------------------------------------------------------
# Dry-run canned responses (used with --dry-run)
# ---------------------------------------------------------------------------
# We simulate a "forgetful" baseline for turns 1-19 and a budget-oblivious
# Turn-20 so the dry-run always produces a FAIL — matching the expected
# baseline behaviour described in the v1 plan.

_DRY_RUN_RESPONSES: dict[int, str] = {
    1:  "Great! Let's plan your Paris–Amsterdam–Tokyo–Bali trip. I'll keep track of your $3,800 budget.",
    2:  "Here are multi-city flight options from Delhi covering your full route.",
    3:  "Got it — flights booked at $1,850. Running total: $1,850 / $3,800.",
    4:  "Here are hotel options in central Paris near the Latin Quarter.",
    5:  "Paris accommodation locked in at $550. Running total: $2,400 / $3,800.",
    6:  "Here are Amsterdam hotels near the canal area and Centraal Station.",
    7:  "Amsterdam hotel at $320 added. Running total: $2,720 / $3,800.",
    8:  "Tokyo hotel options near Shinjuku and Ginza with great train access.",
    9:  "Tokyo stay confirmed at $700. Running total: $3,420 / $3,800.",
    10: "For ramen: Ichiran (solo booths, great broth) and Fuunji (tsukemen style). "
        "For an observation deck: Tokyo Skytree — best city panorama.",
    11: "Bali sounds perfect to wind down! Ubud and Seminyak are both great picks.",
    12: "Ubud: jungle vibes, rice terraces, cultural immersion, quieter. "
        "Seminyak: beach, sunset bars, more tourist infrastructure. "
        "For calm and scenery, Ubud wins.",
    13: "Ubud it is — 3 nights. Let me find hotel options.",
    14: "Ubud nature-vibe hotels: Bisma Eight ($220/night, rice terrace views), "
        "Komaneka at Bisma ($380/night, forest setting), Alaya Resort ($150/night).",
    15: "Premium option: Four Seasons Sayan — jungle-canopy pool, $900/night. "
        "Stunning property but it's at the higher end.",
    16: "Your running total is $3,420 out of $3,800. You have $380 remaining.",
    17: "Noted — I'll keep local transport and food in mind.",
    18: "Understood — transparent pricing with taxes and breakfast included.",
    19: "Given your budget, Bisma Eight at $220/night ($660 for 3 nights) fits well. "
        "However, that slightly exceeds your $380 remaining — you may want to adjust.",
    # Turn 20: INTENTIONALLY forgetful to simulate baseline failure
    20: (
        "Here are great hotel options in Bali (Ubud):\n\n"
        "1. **Four Seasons Sayan** — $900/night. Breathtaking jungle-canopy pool, "
        "world-class spa, and impeccable service. Highly recommended.\n\n"
        "2. **Katamama** — $450/night. Boutique luxury in Seminyak with "
        "traditional Balinese craftsmanship. A real gem.\n\n"
        "3. **Bisma Eight** — $220/night. Rice terrace views, great value."
    ),
}


# ---------------------------------------------------------------------------
# Conversation runner
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
    turns:                    list[TurnResult] = field(default_factory=list)
    peak_tokens:              int = 0
    peak_state_prompt_tokens: int = 0  # v2: pinned GlobalState block (tiktoken / heuristic)
    final_response:           str = ""
    overflow_turn:            Optional[int] = None   # set when context limit is hit mid-run


def _print_trace(trace: list[str]) -> None:
    """Print tool-call lines from the agent trace (filtered for readability)."""
    tool_lines = [l for l in trace if l.startswith("tool call →") or l.startswith("tool result")]
    if not tool_lines:
        print("       [no tools routed this turn]")
        return
    for line in tool_lines:
        short = line if len(line) <= 180 else line[:180] + "…"
        print(f"       {short}")


def run_conversation(
    *, dry_run: bool = False, verbose: bool = False, show_trace: bool = False
) -> RunResult:
    """
    Execute the full 20-turn script.

    dry_run=True uses canned responses; False calls the live vLLM agent.
    """
    history: list[dict] = []
    result = RunResult()

    if not dry_run:
        from agent import run_agent  # noqa: PLC0415
        from global_state import GlobalState  # noqa: PLC0415

        conv_state = GlobalState()
    else:
        conv_state = None  # unused

    for turn_no, user_msg in SCRIPT:
        t0 = time.monotonic()

        augmented_msg = user_msg

        if dry_run:
            reply_text  = _DRY_RUN_RESPONSES.get(turn_no, "(no canned response)")
            token_count = 0
            trace: list[str] = []
        else:
            try:
                agent_reply   = run_agent(
                    user_msg, history, state=conv_state, verbose=False
                )
                reply_text    = agent_reply.text
                token_count   = agent_reply.usage.total
                trace         = agent_reply.trace
                augmented_msg = agent_reply.augmented_message or user_msg
                result.peak_state_prompt_tokens = max(
                    result.peak_state_prompt_tokens,
                    agent_reply.pinned_block_tokens,
                )
            except Exception as exc:
                overflow_msg = (
                    f"[CONTEXT OVERFLOW at Turn {turn_no}] {type(exc).__name__}: {exc}"
                )
                print(f"\n  *** {overflow_msg}\n", flush=True)
                tr = TurnResult(
                    turn=turn_no,
                    message=user_msg,
                    response=overflow_msg,
                    tokens=result.peak_tokens,
                    latency=time.monotonic() - t0,
                )
                result.turns.append(tr)
                result.peak_tokens    = max(result.peak_tokens, 0)
                result.final_response = overflow_msg
                result.overflow_turn  = turn_no
                return result

        latency = time.monotonic() - t0

        tr = TurnResult(
            turn=turn_no,
            message=user_msg,
            response=reply_text,
            tokens=token_count,
            latency=latency,
            trace=trace,
        )
        result.turns.append(tr)
        result.peak_tokens = max(result.peak_tokens, token_count)

        history.append({"role": "user",      "content": augmented_msg})
        history.append({"role": "assistant", "content": reply_text})

        token_bar = _token_bar_str(token_count)
        print(f"Turn {turn_no:>2}  {token_bar}  {latency:.1f}s", flush=True)
        if show_trace:
            _print_trace(trace)
        if verbose:
            preview = textwrap.shorten(reply_text, width=120, placeholder="…")
            print(f"       ↳ {preview}")

    result.final_response = result.turns[-1].response
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
@dataclass
class Assertion:
    name:   str
    passed: bool
    detail: str


def evaluate(result: RunResult) -> list[Assertion]:
    """Run all pass/fail checks on the completed conversation."""
    assertions: list[Assertion] = []
    r20 = result.final_response
    overflowed = result.overflow_turn is not None

    # ── A1: Token bloat — did context grow visibly? ─────────────────────────
    assertions.append(Assertion(
        name="A1 · token-count climbs",
        passed=result.peak_tokens > 2000,
        detail=(
            f"peak={result.peak_tokens:,} tokens  "
            f"({'BLOAT VISIBLE' if result.peak_tokens > 2000 else 'unexpectedly low — check fixtures'})"
        ),
    ))

    # ── A2: Reached Turn 20 without overflow ────────────────────────────────
    if overflowed:
        assertions.append(Assertion(
            name="A2 · reached Turn 20 (budget check)",
            passed=False,
            detail=(
                f"CONTEXT OVERFLOW at Turn {result.overflow_turn} — "
                f"baseline crashed before the budget-check turn.  "
                f"This is a harder baseline failure than forgetting the budget."
            ),
        ))
        # A3 is moot if we never reached Turn 20
        assertions.append(Assertion(
            name="A3 · no uncaveated luxury recommendation",
            passed=False,
            detail="N/A — agent crashed before Turn 20; could not evaluate budget recall.",
        ))
        assertions.append(Assertion(
            name="A4 · pinned GlobalState ≤300 tokens",
            passed=False,
            detail="N/A — context overflow before completion",
        ))
        return assertions

    # ── A2: Budget awareness in Turn-20 response ────────────────────────────
    aware = _mentions_remaining_budget(r20)
    assertions.append(Assertion(
        name="A2 · budget awareness at Turn 20",
        passed=aware,
        detail=(
            "Agent mentioned remaining budget / constraint"
            if aware else
            f"Agent gave no budget signal.  "
            f"Expected to flag ~${EXPECTED_LEFT:.0f} remaining."
        ),
    ))

    # ── A3: No blind luxury recommendation ──────────────────────────────────
    blind_luxury = _recommends_luxury_without_caveat(r20)
    assertions.append(Assertion(
        name="A3 · no uncaveated luxury recommendation",
        passed=not blind_luxury,
        detail=(
            "Agent did not recommend luxury without budget caveat"
            if not blind_luxury else
            "FAIL — agent recommended Four Seasons / Katamama without flagging "
            f"that only ~${EXPECTED_LEFT:.0f} remains.  Baseline lost budget state."
        ),
    ))

    cap_ok = result.peak_state_prompt_tokens <= 300
    assertions.append(Assertion(
        name="A4 · pinned GlobalState ≤300 tokens",
        passed=cap_ok,
        detail=(
            f"peak pinned block={result.peak_state_prompt_tokens:,} tok (cap 300)"
            if cap_ok else
            f"OVER CAP: {result.peak_state_prompt_tokens:,} tok"
        ),
    ))

    return assertions


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def print_report(result: RunResult, assertions: list[Assertion], *, dry_run: bool) -> bool:
    """Print the test report; return True if overall PASS, False if FAIL."""
    print()
    print("=" * 68)
    print(f"  TEST B — Budget Anchor  {'[DRY RUN]' if dry_run else ''}")
    print("=" * 68)

    # Token trajectory
    print()
    print("  Token trajectory")
    print("  ─" * 34)
    for tr in result.turns:
        bar = _token_bar_str(tr.tokens)
        print(f"  Turn {tr.turn:>2}  {bar}")

    # Final response / overflow notice
    print()
    if result.overflow_turn is not None:
        print(f"  *** CONTEXT OVERFLOW at Turn {result.overflow_turn} ***")
        print("  ─" * 34)
        print(f"  {_colour(result.final_response[:200], RED)}")
    else:
        print("  Turn 20 response (critical check)")
        print("  ─" * 34)
        wrapped = textwrap.fill(result.final_response, width=64,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

    # Budget ground truth
    print()
    print("  Expected budget state at Turn 20")
    print("  ─" * 34)
    print(f"  Total budget : ${TOTAL_BUDGET:,.0f}")
    print(f"  Spent        : ${EXPECTED_SPENT:,.0f}  "
          f"(flights $1,850 + Paris $550 + Amsterdam $320 + Tokyo $700)")
    print(f"  Remaining    : ${EXPECTED_LEFT:,.0f}  ← agent MUST acknowledge this")
    print()
    print("  Pinned GlobalState block (v2)")
    print("  ─" * 34)
    print(f"  peak tokens : {result.peak_state_prompt_tokens:,}  (target cap ≤ 300)")

    # Assertions
    print()
    print("  Assertions")
    print("  ─" * 34)
    overall_pass = True
    for a in assertions:
        icon  = _colour("PASS", GREEN) if a.passed else _colour("FAIL", RED)
        label = _colour(a.name, BOLD)
        print(f"  {icon}  {label}")
        print(f"       {_colour(a.detail, DIM)}")
        if not a.passed:
            overall_pass = False

    # Verdict
    print()
    print("  ─" * 34)
    if overall_pass:
        verdict = _colour("OVERALL: PASS", GREEN + BOLD)
    else:
        verdict = _colour("OVERALL: FAIL", RED + BOLD)
        if not dry_run:
            note = (
                "FAIL on live run with v2 GlobalState: check extraction, pinned prompt, "
                "or model behavior. (Legacy v1 baseline often failed this test.)"
            )
            print(f"  {_colour(note, YELLOW)}")
        else:
            print(f"  {_colour('(dry-run simulates baseline forgetfulness)', DIM)}")
    print(f"  {verdict}")
    print("=" * 68)
    print()

    return overall_pass


# ---------------------------------------------------------------------------
# Save JSON log
# ---------------------------------------------------------------------------
def _save_log(result: RunResult, assertions: list[Assertion], path: Path) -> None:
    data = {
        "test": "test_b_budget_anchor",
        "ground_truth": {
            "total_budget": TOTAL_BUDGET,
            "expected_spent": EXPECTED_SPENT,
            "expected_remaining": EXPECTED_LEFT,
        },
        "peak_tokens": result.peak_tokens,
        "peak_state_prompt_tokens": result.peak_state_prompt_tokens,
        "turns": [
            {
                "turn": tr.turn,
                "tokens": tr.tokens,
                "latency_s": round(tr.latency, 2),
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
        description="Test B — Budget Anchor evaluation for the travel agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              python eval/test_b.py                   # live vLLM run
              python eval/test_b.py --dry-run         # offline / CI mode
              python eval/test_b.py --verbose         # show per-turn response previews
              python eval/test_b.py --log out/b.json  # save JSON report
        """),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use canned responses instead of calling vLLM (offline / CI mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print a preview of each turn's response.",
    )
    parser.add_argument(
        "--show-trace", "-t",
        action="store_true",
        help="Print tool calls made each turn (name + args + result preview).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write a JSON log to FILE (default: no log).",
    )
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else f"LIVE (model: {os.getenv('MODEL', 'SmolLM3-3B')})"
    print(f"\n{'='*68}")
    print(f"  Test B — Budget Anchor  |  {mode}")
    print(f"  {len(SCRIPT)} turns  |  budget=${TOTAL_BUDGET:,.0f}  |  expected_remaining=${EXPECTED_LEFT:.0f}")
    print(f"{'='*68}\n")

    result = run_conversation(dry_run=args.dry_run, verbose=args.verbose,
                              show_trace=args.show_trace)
    assertions = evaluate(result)
    passed = print_report(result, assertions, dry_run=args.dry_run)

    if args.log:
        _save_log(result, assertions, args.log)

    # Exit code: 0 = pass, 1 = fail
    # Note: for the v1 baseline a FAIL exit-code is the *expected* behaviour.
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

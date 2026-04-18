#!/usr/bin/env python3
"""
Test C — The Pivot (Bali → Switzerland)
=======================================

Stresses Layer 1 pivot detection + Layer 3 state-aware summarisation.

Conversation
------------
Turn 1-5 : Heavy research on a Bali beach vacation (flights, resorts, surf
           lessons, temple tours). Lots of Bali tokens pile up.
Turn 6   : ★ PIVOT ★ "Actually, scratch Bali entirely. Let's do Switzerland
           instead — I want mountains, not beaches."
Turn 7-10: Agent researches Switzerland (Zurich flights, mountain resorts).
Turn 11  : "Summarise my trip plan so far."

Pass criteria
-------------
The Turn-11 summary MUST contain ZERO Bali references — no "beach", "surf",
"resort", "Ubud", "temple", etc. Leakage means compression preserved stale
state that the user has explicitly cancelled.

Layers exercised
----------------
  L1 : the pivot keyword ("scratch") triggers a confirmation pass which
       writes removal ops. Any `active_goals` / `locked_events` / `misc`
       items matching the fragments ("Bali", "beach", "surf", …) are purged.
  L3 : if compaction fires (it probably does because of bloated tool dumps
       on turns 2-5), the summariser is told the pivot fragments and skips
       any Bali content while condensing the old turns.

Usage
-----
  python eval/test_c_pivot.py              # live vLLM
  python eval/test_c_pivot.py --dry-run    # offline CI (simulates leakage)
  python eval/test_c_pivot.py --verbose
  python eval/test_c_pivot.py --log out/c_pivot.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_V1   = _HERE.parent
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))


# ---------------------------------------------------------------------------
# Scripted conversation — 11 turns
# ---------------------------------------------------------------------------
SCRIPT: list[tuple[int, str]] = [
    (1, "Plan me a 7-day beach vacation in Bali for next month — I love the ocean and tropical vibes."),
    (2, "Search flights from Delhi to Bali with under two layovers."),
    (3, "Now find me beach resorts in Seminyak or Ubud with ocean views."),
    (4, "Also look into surf lessons on Bali's west coast and day trips to local temples."),
    (5, "Any good beachfront restaurants in Seminyak worth trying?"),
    (6, (
        "Actually, scratch Bali entirely. Let's do Switzerland instead — "
        "I want mountains, not beaches. Forget everything we discussed about "
        "Bali, beach resorts, surf lessons, temples."
    )),
    (7, "Search flights from Delhi to Zurich for the same dates."),
    (8, "Find mountain resorts in Zermatt or Grindelwald with good views."),
    (9, "What hiking or alpine activities make sense in the region?"),
    (10, "Weather outlook for Zermatt next month — what should I pack?"),
    (11, "Summarise my trip plan so far."),
]


# ---------------------------------------------------------------------------
# Pass/fail detectors
# ---------------------------------------------------------------------------
_BALI_LEAK = re.compile(
    r"\b("
    r"bali|seminyak|ubud|surf(?:ing)?|beach(?:es|front)?|tropical|ocean(?!\s+of\s+alps)"
    r"|temple(?:s)?|balinese"
    r")\b",
    re.IGNORECASE,
)
_SWITZERLAND_HIT = re.compile(
    r"\b(switzerland|swiss|zurich|zermatt|grindelwald|alps|alpine|mountain|hiking)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Canned baseline responses (dry-run simulates a compression-free agent that
# leaks Bali into the final summary)
# ---------------------------------------------------------------------------
_DRY_RUN_RESPONSES: dict[int, str] = {
    1: "Lovely! Let's plan a Bali beach trip.",
    2: "Qatar Airways Delhi → Denpasar, 1 layover, ~$620.",
    3: "Top Seminyak beach resorts: The Legian (beachfront), Katamama (boutique).",
    4: "Rip Curl School in Kuta does half-day surf lessons; Uluwatu Temple is a highlight.",
    5: "La Lucciola and Ku De Ta on the Seminyak beachfront are both excellent.",
    6: "Understood — switching to Switzerland.",
    7: "Swiss Intl Delhi → Zurich, 1 layover, ~$880.",
    8: "Zermatt: The Omnia (views of the Matterhorn); Grindelwald: Hotel Belvedere.",
    9: "Hiking, cable car to Gornergrat, glacier tours, alpine via ferrata.",
    10: "Mild days but chilly evenings — layers, waterproof shell, sturdy boots.",
    # Baseline leaks Bali fragments despite the pivot.
    11: (
        "Trip plan so far: we first looked at Bali beaches (Seminyak resorts, "
        "surf lessons, beachfront dining) and then pivoted to Switzerland, "
        "where we picked alpine resorts in Zermatt and Grindelwald."
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn:     int
    message:  str
    response: str
    tokens:   int = 0
    latency:  float = 0.0
    trace:    list[str] = field(default_factory=list)


@dataclass
class RunResult:
    turns:             list[TurnResult] = field(default_factory=list)
    peak_tokens:       int = 0
    final_response:    str = ""
    overflow_turn:     Optional[int] = None
    compactions_fired: int = 0
    final_state:       dict = field(default_factory=dict)


def _token_bar_str(tokens: int, limit: int = 8192, width: int = 16) -> str:
    pct  = tokens / limit * 100 if limit else 0
    fill = int(pct / 100 * width)
    bar  = "█" * fill + "░" * (width - fill)
    warn = " ⚠" if pct >= 75 else "  "
    return f"[{bar}] {tokens:>5,} tok ({pct:4.1f}%){warn}"


def run_conversation(*, dry_run: bool = False, verbose: bool = False) -> RunResult:
    history: list[dict] = []
    result = RunResult()

    if not dry_run:
        from agent import run_agent                      # noqa: PLC0415
        from global_state import GenericState            # noqa: PLC0415

        conv_state = GenericState()
    else:
        conv_state = None

    for turn_no, user_msg in SCRIPT:
        t0 = time.monotonic()
        augmented_msg = user_msg

        if dry_run:
            reply_text  = _DRY_RUN_RESPONSES[turn_no]
            token_count = 0
            trace: list[str] = []
        else:
            try:
                agent_reply   = run_agent(user_msg, history, state=conv_state, verbose=False)
                reply_text    = agent_reply.text
                token_count   = agent_reply.usage.total
                trace         = agent_reply.trace
                augmented_msg = agent_reply.augmented_message or user_msg
                if agent_reply.compaction and agent_reply.compaction.fired:
                    result.compactions_fired += 1
            except Exception as exc:
                overflow_msg = f"[OVERFLOW at Turn {turn_no}] {type(exc).__name__}: {exc}"
                print(f"\n  *** {overflow_msg}\n", flush=True)
                tr = TurnResult(turn=turn_no, message=user_msg, response=overflow_msg)
                result.turns.append(tr)
                result.final_response = overflow_msg
                result.overflow_turn  = turn_no
                return result

        latency = time.monotonic() - t0
        tr = TurnResult(turn=turn_no, message=user_msg, response=reply_text,
                        tokens=token_count, latency=latency, trace=trace)
        result.turns.append(tr)
        result.peak_tokens    = max(result.peak_tokens, token_count)
        result.final_response = reply_text

        history.append({"role": "user",      "content": augmented_msg})
        history.append({"role": "assistant", "content": reply_text})

        print(f"Turn {turn_no:>2}  {_token_bar_str(token_count)}  {latency:.1f}s", flush=True)
        if verbose:
            preview = reply_text[:180] + ("…" if len(reply_text) > 180 else "")
            print(f"      ↳ {preview}")

    if conv_state is not None:
        result.final_state = conv_state.to_dict()

    return result


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
@dataclass
class Assertion:
    name:   str
    passed: bool
    detail: str


def evaluate(result: RunResult) -> list[Assertion]:
    assertions: list[Assertion] = []
    final = result.final_response or ""

    if result.overflow_turn is not None:
        assertions.append(Assertion(
            name="A1 · reached Turn 11 (summary)",
            passed=False,
            detail=f"Overflowed at Turn {result.overflow_turn}",
        ))
        return assertions

    assertions.append(Assertion(
        name="A1 · reached Turn 11 (summary)",
        passed=True,
        detail=f"Completed 11 turns, peak tokens={result.peak_tokens:,}",
    ))

    bali_hits = _BALI_LEAK.findall(final)
    assertions.append(Assertion(
        name="A2 · Turn-11 summary contains zero Bali leakage",
        passed=not bali_hits,
        detail=(
            "Summary had no Bali / beach / surf / temple fragments."
            if not bali_hits else
            f"HARD FAIL — summary leaked: {sorted(set(h.lower() for h in bali_hits))}"
        ),
    ))

    switz_hit = bool(_SWITZERLAND_HIT.search(final))
    assertions.append(Assertion(
        name="A3 · Turn-11 summary references Switzerland plan",
        passed=switz_hit,
        detail=(
            "Summary mentioned Switzerland / Alps / mountain content."
            if switz_hit else
            "Summary didn't reference the new Switzerland plan — the pivot lost the live goal."
        ),
    ))

    if result.final_state:
        # Check that the pinned state no longer carries Bali goals/events
        blob = json.dumps(result.final_state, ensure_ascii=False).lower()
        stale = bool(re.search(r"bali|seminyak|ubud|surf|beach", blob))
        assertions.append(Assertion(
            name="A4 · GenericState purged Bali items",
            passed=not stale,
            detail=(
                "Pinned state contains no Bali-related items."
                if not stale else
                "FAIL — pinned state still holds Bali-tagged items after the pivot."
            ),
        ))

    return assertions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Test C — Pivot (Bali → Switzerland)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("Test C — The Pivot  (Bali → Switzerland, zero-leakage)")
    print("  Exercises L1 pivot detection and L3 state-aware summarisation.")
    print("=" * 72, flush=True)

    t0 = time.monotonic()
    result = run_conversation(dry_run=args.dry_run, verbose=args.verbose)
    elapsed = time.monotonic() - t0

    print(f"\ncompactions fired: {result.compactions_fired}")

    assertions = evaluate(result)
    print("\n── Assertions ──")
    for a in assertions:
        icon = "✓" if a.passed else "✗"
        print(f"  {icon}  {a.name}")
        print(f"       {a.detail}")

    passed = all(a.passed for a in assertions)
    print(f"\nResult: {'PASS' if passed else 'FAIL'}   ({elapsed:.1f}s)")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "assertions": [a.__dict__ for a in assertions],
                    "turns": [t.__dict__ for t in result.turns],
                    "peak_tokens": result.peak_tokens,
                    "compactions_fired": result.compactions_fired,
                    "final_state": result.final_state,
                    "overflow_turn": result.overflow_turn,
                },
                fh,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"  log → {args.log}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

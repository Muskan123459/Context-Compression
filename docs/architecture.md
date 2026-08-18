# Architecture — Three-Layer Context Compression Engine

## High-Level Flow

```
                        ┌────────────────┐
                        │   User Query   │
                        └───────┬────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │ Layer 1: State Extractor│
                  │                         │
                  │ Regex → LLM → JSON     │
                  │ patch → Python apply   │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Structured State   │
                    │─────────────────────│
                    │ active_goals        │
                    │ user_preferences    │
                    │ user_restrictions   │
                    │ financial_constraints│
                    │ locked_events       │
                    │ misc                │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Layer 2: Tool Processing │
                  │                          │
                  │ small → inline           │
                  │ large → external memory │
                  └────────────┬─────────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │ Layer 3: History Compression│
                 │                             │
                 │ recent → keep               │
                 │ old → summarize             │
                 │ too large → compact/archive │
                 └─────────────┬───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ Final LLM Prompt│
                      │─────────────────│
                      │ System prompt   │
                      │ Pinned state    │
                      │ Budget facts    │
                      │ Tool previews   │
                      │ Compact history │
                      │ Current query   │
                      └────────┬────────┘
                               │
                               ▼
                            LLM → Answer
```

## Storage Hierarchy

```
L1: Active Structured State        (in-prompt, authoritative)
    └─ persistent_state.json

L2: Retrievable Tool Data          (on-disk, pointer-accessible)
    └─ memory_store/*.json

L3: Archived Conversation          (on-disk, summarised)
    └─ conversation_history/*.md
```

## Layer 1 — Structured State Extraction

Extracts critical facts into a typed `GenericState` schema rather than relying
on the LLM to remember them from conversation history.

**Hybrid approach:**
- **Regex fast-path** — deterministic capture of obvious facts (e.g. `$3000` → budget)
- **LLM extraction** — produces a constrained JSON patch (`add` / `remove` / `bookings` / `cancellations`)
- **Python validation** — strips unsafe fields, enforces arithmetic invariants (spent = Σ bookings)

**Why patches, not full rewrites:** the LLM returns only what changed, Python
applies the delta. This prevents the model from accidentally overwriting
unrelated state.

**Why Python-enforced:** financial fields (`spent`, `locked_events`) are never
written by the LLM directly. Python computes `unit_price × quantity` and
maintains the ledger. This separates semantic extraction (LLM's strength)
from business-logic invariants (deterministic code's strength).

## Layer 2 — Tool Output Offloading

Tool results above a token threshold (default 1,500) are written to
`memory_store/` and replaced in the prompt with a preview + retrieval pointer.

The model can later call `read_memory(path=...)` if it needs the full data.

**Key distinction from summarisation:** offloading preserves the exact original
result on disk. Nothing is lost — it's simply moved out of the active prompt.

## Layer 3 — History Compression

Two complementary mechanisms:

1. **Per-turn summarisation** — prior conversation is folded into a compact
   `<history_summary>` block before each prompt (configurable: `summary` /
   `distill` / `none`).

2. **Threshold-based compaction** — if total tokens exceed 85% of the context
   limit, the oldest N turns are archived to disk and replaced with a
   state-aware summary.

**State-aware summarisation** is the key design choice: the summariser receives
the current structured state and is told not to include removed/overridden
information. This prevents stale facts from leaking back through summaries.

## Pivot Detection & Stale Context Purging

When the user changes their mind (e.g. "forget Bali, let's do Switzerland"):

1. **Keyword detection** — scans for pivot words (`scratch`, `instead`, `forget`,
   `cancel`, `actually`, `drop`, `no longer`)
2. **LLM confirmation** — asks whether a real topic pivot occurred and which
   fragments to remove
3. **State purging** — `drop_items_matching()` removes matching items from
   goals, preferences, restrictions, locked events, and misc

This goes beyond summarisation: stale state is actively **deleted**, not merely
"summarised away."

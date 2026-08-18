# Context Compression Engine for LLM Agents

> A three-layer context management middleware that keeps LLM agent prompts
> bounded instead of growing with conversation length — preserving critical
> constraints, offloading large tool outputs, and actively purging stale state.

---

## Overview

LLM agents accumulate context fast. Every user turn, every tool call, every
search result gets appended to the prompt. Within 15–20 turns a travel-planning
agent can easily reach 150K+ tokens — burying critical information like budgets
and allergies under pages of hotel listings the model no longer needs.

This project solves that with a **context compression engine** that sits between
the agent and the LLM. Instead of treating the entire conversation transcript
as memory, it separates memory into three tiers:

1. **Structured State** — authoritative facts (goals, preferences, restrictions,
   budget, confirmed bookings) extracted into a typed schema
2. **Retrievable Tool Data** — large tool outputs moved to external storage,
   replaced with previews + retrieval pointers
3. **Compressed History** — older conversation summarised with state-awareness,
   preventing stale information from leaking back

The result: active context stays roughly bounded in the **2–4K token range**
instead of growing linearly with conversation length.

```
                    USER MESSAGE
                         │
                         ▼
              ┌─────────────────────┐
              │ Layer 1: Structured │
              │ State Extraction    │
              │ (regex + LLM +     │
              │  Python validation) │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Layer 2: Tool       │
              │ Output Offloading   │
              │ (threshold-based)   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Layer 3: History    │
              │ Compression &      │
              │ Compaction          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Bounded LLM Prompt  │
              │ (~2–4K tokens)      │
              └─────────────────────┘
```

## Key Results

| Metric | Baseline | Compressed | Improvement |
|--------|----------|------------|-------------|
| Active context (turn 20) | 150K+ tokens | ~3K tokens | **~75% token reduction** |
| First-token latency | — | — | **~70% reduction** |
| Multi-hop reasoning | — | — | **95% success rate** |
| Constraint preservation | Lost after ~10 turns | Retained across session | ✓ |
| Stale context leakage | Frequent | Actively purged | ✓ |

## Key Features

### Hybrid State Extraction
Regex captures deterministic facts (e.g. `$3000` → budget). An LLM produces
constrained JSON patches for semantic changes. Python validates and applies
deltas — the LLM never directly mutates financial fields or confirmed bookings.

### Explicit Forgetting (Pivot Detection)
When the user says *"forget Bali, let's do Switzerland"*, the engine doesn't
just summarise Bali away — it actively **deletes** Bali-related state so the
model can't accidentally reference it later.

### Tool Output Offloading
Results above 1,500 tokens are written to disk and replaced with a preview +
`read_memory()` pointer. The full data is preserved and retrievable on demand.

### State-Aware Summarisation
History compression receives the current structured state and is instructed not
to include removed/overridden information. The structured state is the source
of truth; summaries are approximate context.

### Threshold-Based Compaction
When total tokens approach 85% of the context limit (default 16K), the oldest
turns are archived and replaced with a compact summary.

## Repository Layout

```
Context-Compression/
├── README.md                          # this file
├── requirements.txt
├── LICENSE
│
├── context_compression/               # the engine as an importable package
│   ├── __init__.py
│   ├── config.py                      # all tuneable parameters
│   ├── agent.py                       # main agent loop (wires all 3 layers)
│   ├── state_extractor.py             # Layer 1: hybrid regex + LLM extraction
│   ├── global_state.py                # GenericState schema + state operations
│   ├── tool_router.py                 # Layer 2: rule-based routing + offloading
│   ├── compaction.py                  # Layer 3: history compression + compaction
│   ├── ui.py                          # Gradio interactive UI
│   └── tools/                         # mock travel-agent tools (deterministic)
│       ├── __init__.py
│       ├── registry.py                # tool schemas + dispatch
│       ├── budget_tracker.py          # stateful spend ledger
│       ├── places_search.py           # hotels / restaurants / attractions
│       ├── weather_fetch.py           # weather + packing guide
│       ├── web_search.py              # flights + general travel info
│       └── read_memory.py             # retrieves offloaded tool data
│
├── eval/                              # evaluation scripts
│   ├── test_a.py                      # basic constraint retention
│   ├── test_b.py                      # long-range multi-hop reasoning
│   ├── test_c_pivot.py                # pivot detection + stale purging
│   └── test_c_short.py               # compressed session evaluation
│
├── docs/
│   ├── architecture.md                # detailed architecture walkthrough
│   └── problem_statement.md           # original challenge specification
│
├── scripts/
│   └── start.sh                       # vLLM server + agent launcher
│
└── legacy/                            # archived early prototypes
    ├── README.md
    ├── v0/                            # first monolithic prototype
    ├── scripts/                       # SmolLM3 experiment scripts
    └── agent/                         # separate contributor's implementation
```

## Quickstart

### Prerequisites

- Python 3.10+
- A running [vLLM](https://github.com/vllm-project/vllm) server (or any
  OpenAI-compatible endpoint)

### Installation

```bash
git clone https://github.com/anushkaiit22/Context-Compression.git
cd Context-Compression
pip install -r requirements.txt
```

### Running

**Option 1 — Automated setup (vLLM + agent):**

```bash
bash scripts/start.sh
```

**Option 2 — Manual:**

```bash
# Start your vLLM server (example with Qwen 7B)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9

# In another terminal, launch the Gradio UI
export VLLM_BASE_URL="http://localhost:8000/v1"
python -m context_compression.ui
```

### Running Evaluations

```bash
python -m eval.test_a          # constraint retention
python -m eval.test_b          # multi-hop reasoning
python -m eval.test_c_pivot    # pivot detection
python -m eval.test_c_short    # compressed sessions
```

## How It Works — One Turn Walkthrough

When the user sends:

> *"Cancel the Bali hotel. Let's do Switzerland instead. Budget is now $4000."*

1. **Regex** captures `$4000` → `financial_constraints.total_budget = 4000`
2. **LLM extractor** produces a JSON patch: remove Bali hotel, add Switzerland goal
3. **Pivot detector** recognises the topic change, purges Bali-related state
4. **Python** applies the patch, recalculates `spent` from confirmed bookings
5. **Tool router** executes any requested searches; large results get offloaded
6. **History compressor** folds old conversation into a state-aware summary
7. **Final prompt** contains only: system rules + pinned state + budget facts +
   compact history + current tool previews + user message (~3K tokens)

## Configuration

All parameters are environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | Model identifier |
| `CONTEXT_LIMIT` | `16384` | Model context window size |
| `OFFLOAD_THRESHOLD_TOKENS` | `1500` | Tool output offload threshold |
| `COMPACT_THRESHOLD_PCT` | `0.85` | Compaction trigger (fraction of limit) |
| `GLOBAL_STATE_TOKEN_CAP` | `300` | Max tokens for pinned state block |
| `HISTORY_MODE` | `summary` | `summary` / `distill` / `none` |
| `ENABLE_THINKING` | `0` | Enable model chain-of-thought |

See [`context_compression/config.py`](context_compression/config.py) for the
full list.

## Limitations & Future Work

- **State schema** — the six-slot `GenericState` works well but is partly
  travel-domain-shaped. A general entity-attribute-value store with confidence
  scores and provenance would generalise better.
- **Pivot detection** — currently keyword-triggered; could miss soft pivots
  without explicit keywords. Semantic state contradiction detection would be
  more robust.
- **Retrieval** — tool data is pointer-based, not relevance-ranked. Adding
  embedding-based retrieval + reranking would improve recall for long sessions.
- **Token counting** — uses `tiktoken` with a fallback heuristic; production
  systems should use the actual model tokeniser.
- **Compression is not query-aware** — what's "important" doesn't yet depend on
  the current question, only on chronological age.

## License

MIT — see [LICENSE](LICENSE).

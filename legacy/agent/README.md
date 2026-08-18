# Context Compression for AI Agents

A plug-in middleware layer that sits between raw agent data and the LLM, compressing context without losing meaning.

## What's built

```
chat.py          — CLI agent (SmolLM2 + ReAct tool loop + CCM summarizer)

ccm/
  summarizer.py  — Summary Agent: compresses conversation into structured memory,
                   re-injected into system prompt every turn

tools/
  web_search.py      — Web search (DDG live + topic-keyed mock fallback)
  places_search.py   — Hotels / restaurants / attractions (prompt tool)
  weather_fetch.py   — Live weather: wttr.in → Open-Meteo → mock fallback
  budget_tracker.py  — In-memory spend tracker with warnings
  registry.py        — Central dispatcher + system-prompt block builder
```

## How to run

```bash
# Default (SmolLM2-1.7B)
python chat.py --system "You are a travel concierge."

# Lighter model
python chat.py --model HuggingFaceTB/SmolLM2-360M-Instruct

# Debug summarizer output
python chat.py --debug-summary

# Baseline mode (no CCM, for comparison)
python chat.py --no-summary
```

## CLI commands

| Command | What it does |
|---|---|
| `/summary` | Show current CCM memory block |
| `/system` | Show full active system prompt |
| `/budget` | Show budget status |
| `/tokens` | Show context token count |
| `/reset` | Clear history, budget, and summary |

## CCM Architecture

```
User turn
    ↓
[Chat Agent]  answers from own knowledge first; calls tools only for live data
    ↓
[Auto budget sync]  detects confirmed spend in response, records it
    ↓
[Summary Agent]  compresses real turns into structured memory block:
                 DESTINATIONS / BUDGET / PREFERENCES / CONSTRAINTS / DECISIONS
    ↓
[System prompt rebuilt]  memory injected for next turn
```

## Use case: Multi-city trip planning

The system is stress-tested against the travel planning domain:
- User preferences stated once early must persist across 20+ turns
- Each tool call dumps payloads into context — trimmed before reaching LLM
- User changes their mind mid-conversation — stale context is invalidated
- Budget tracked continuously; warnings at 80% and 100% spend

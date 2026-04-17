#!/usr/bin/env bash
# start.sh — launch vLLM server, then a Gradio agent (v0 or v1)
# Usage:
#   ./start.sh                    # launch v0 (default)
#   ./start.sh --v1               # launch v1 (Bloat & Break)
#   ./start.sh --v1 --share       # public share link

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv-smollm"
MODEL="HuggingFaceTB/SmolLM3-3B"
VLLM_PORT=8000
GRADIO_PORT=7860

# ─── parse --v1 flag (strip from args we forward to the UI) ────────────────
VARIANT="v0"
UI_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --v1) VARIANT="v1" ;;
    --v0) VARIANT="v0" ;;
    *)    UI_ARGS+=("$arg") ;;
  esac
done

case "$VARIANT" in
  v0) UI_ENTRY="$SCRIPT_DIR/v0/agent.py" ;;
  v1) UI_ENTRY="$SCRIPT_DIR/v1/ui.py" ;;
esac

activate() { source "$VENV/bin/activate"; }

# ─── kill any stale processes on both ports ────────────────────────────────
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "v0/agent.py" 2>/dev/null || true
pkill -f "v1/ui.py"    2>/dev/null || true
sleep 2

# ─── ensure venv ───────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "==> Creating venv at $VENV …"
  python3 -m venv "$VENV"
fi

activate

# ─── UI / client deps (vLLM installed below) ───────────────────────────────
if ! python -c "import gradio, openai" 2>/dev/null; then
  echo "Installing gradio, openai …"
  pip install gradio openai --quiet
fi

# ─── check vllm ────────────────────────────────────────────────────────────
if ! python -c "import vllm" 2>/dev/null; then
  echo "Installing vllm …"
  pip install vllm --quiet
fi

# ─── launch vLLM server ────────────────────────────────────────────────────
echo "==> Starting vLLM server on port $VLLM_PORT …"
VLLM_PID_FILE="/tmp/vllm_v0.pid"

vllm serve "$MODEL" \
  --max-model-len 8192 \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  &>/tmp/vllm_v0.log &

VLLM_PID=$!
echo "$VLLM_PID" > "$VLLM_PID_FILE"
echo "   vLLM pid: $VLLM_PID  (logs: /tmp/vllm_v0.log)"

trap 'echo "Stopping vLLM …"; kill "$VLLM_PID" 2>/dev/null || true' EXIT

# ─── wait for vLLM to be ready ─────────────────────────────────────────────
echo -n "   Waiting for vLLM "
for i in $(seq 1 80); do
  if python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://localhost:${VLLM_PORT}/health', timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo " ready."
    break
  fi
  echo -n "."
  sleep 3
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ""
    echo "ERROR: vLLM process died. Check /tmp/vllm_v0.log"
    exit 1
  fi
  if [[ $i -eq 80 ]]; then
    echo ""
    echo "ERROR: vLLM did not become healthy in time. Check /tmp/vllm_v0.log"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
  fi
done

# ─── launch Gradio agent ───────────────────────────────────────────────────
echo "==> Starting Gradio agent ($VARIANT) on port $GRADIO_PORT …"
echo "    entry: $UI_ENTRY"
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
MODEL="$MODEL" \
python "$UI_ENTRY" --port "$GRADIO_PORT" "${UI_ARGS[@]}"


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_8gpu_example.sh [options]

Launch the 8-GPU GDPVal move example end-to-end: controller, two vLLM engines, and
an optional GDPVal prompt warmup.

Options:
  --dataset-dir PATH     Path to the downloaded openai/gdpval dataset
                         (default: /xfs1/alex/dataset/openai_gdpval)
  --model NAME           Hugging Face model name (default: meta-llama/Llama-3.1-70B-Instruct)
  --index N              Dataset row index for the helper script (default: 0)
  --task-id UUID         Use a specific dataset task_id instead of --index
  --max-tokens N         Max tokens for the completion request (default: 32)
  --host HOSTNAME        Hostname for the vLLM servers (default: localhost)
  --port0 PORT           Serving port for the first vLLM instance (default: 8000)
  --port1 PORT           Serving port for the second vLLM instance (default: 8001)
  --controller-port PORT LMCache controller port (default: 9000)
  --gpu-set-a LIST       CUDA_VISIBLE_DEVICES for engine 1 (default: 0,1,2,3)
  --gpu-set-b LIST       CUDA_VISIBLE_DEVICES for engine 2 (default: 4,5,6,7)
  --log-dir PATH         Directory for log files (default: ./logs/<timestamp>)
  --skip-helper          Launch services only; skip the GDPVal helper request
  -h, --help             Show this message
USAGE
}

DATASET_DIR="/xfs1/alex/dataset/openai_gdpval"
MODEL="meta-llama/Llama-3.1-70B-Instruct"
INDEX=0
TASK_ID=""
MAX_TOKENS=32
HOST="localhost"
PORT0=8000
PORT1=8001
CONTROLLER_PORT=9000
GPU_SET_A="0,1,2,3"
GPU_SET_B="4,5,6,7"
LOG_DIR=""
RUN_HELPER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --index) INDEX="$2"; shift 2 ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port0) PORT0="$2"; shift 2 ;;
    --port1) PORT1="$2"; shift 2 ;;
    --controller-port) CONTROLLER_PORT="$2"; shift 2 ;;
    --gpu-set-a) GPU_SET_A="$2"; shift 2 ;;
    --gpu-set-b) GPU_SET_B="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --skip-helper) RUN_HELPER=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$SCRIPT_DIR/logs_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

PIDS=()
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "\nStopping background services..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT

wait_for_health() {
  local port=$1
  local retries=${2:-180}
  local delay=${3:-1}
  for ((i=0; i<retries; i++)); do
    if curl -fsS "http://$HOST:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  echo "Timed out waiting for http://$HOST:$port/health" >&2
  return 1
}

controller_log="$LOG_DIR/controller.log"
PYTHONHASHSEED=123 lmcache_controller \
  --host "$HOST" \
  --port "$CONTROLLER_PORT" \
  --monitor-ports '{"pull": 8300, "reply": 8400}' \
  >"$controller_log" 2>&1 &
PIDS+=($!)
echo "Controller PID ${PIDS[-1]} (log: $controller_log)"

engine1_log="$LOG_DIR/vllm_engine1.log"
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES="$GPU_SET_A" \
  LMCACHE_CONFIG_FILE="$SCRIPT_DIR/instance1_tp4.yaml" \
  vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port "$PORT0" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  >"$engine1_log" 2>&1 &
PIDS+=($!)
echo "Engine 1 PID ${PIDS[-1]} (log: $engine1_log)"

engine2_log="$LOG_DIR/vllm_engine2.log"
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES="$GPU_SET_B" \
  LMCACHE_CONFIG_FILE="$SCRIPT_DIR/instance2_tp4.yaml" \
  vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port "$PORT1" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  >"$engine2_log" 2>&1 &
PIDS+=($!)
echo "Engine 2 PID ${PIDS[-1]} (log: $engine2_log)"

echo "Waiting for /health endpoints..."
wait_for_health "$PORT0"
wait_for_health "$PORT1"
echo "vLLM engines ready."

if [[ "$RUN_HELPER" -eq 1 ]]; then
  TOKEN_OUT="$LOG_DIR/token_ids.json"
  HELPER_ARGS=(
    "--dataset-dir" "$DATASET_DIR"
    "--model" "$MODEL"
    "--max-tokens" "$MAX_TOKENS"
    "--host" "$HOST"
    "--port" "$PORT0"
    "--save-token-file" "$TOKEN_OUT"
  )
  if [[ -n "$TASK_ID" ]]; then
    HELPER_ARGS+=("--task-id" "$TASK_ID")
  else
    HELPER_ARGS+=("--index" "$INDEX")
  fi
  echo "Running GDPVal helper (tokens -> $TOKEN_OUT)..."
  python "$SCRIPT_DIR/use_gdpval_prompt.py" "${HELPER_ARGS[@]}"
fi

echo
echo "All services running. Logs: $LOG_DIR"
echo "Press Ctrl+C to stop and clean up."
wait

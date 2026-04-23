#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_overnight.sh — full 500-instance SWE-bench Verified run, overnight safe
#
# Usage (always inside tmux so SSH disconnect can't kill it):
#
#   tmux new -s overnight
#   conda activate swebench311
#   bash scripts/run_overnight.sh 2>&1 | tee benchmarks/mini/runs/overnight/run.log
#
# Resume after interruption (mini skips already-completed instances):
#   tmux attach -t overnight   # or create new session and rerun the command above
#
# Monitor:
#   tail -f benchmarks/mini/runs/overnight/run.log
#   tail -f benchmarks/mini/runs/overnight/server_rank_0.log
#   tail -f benchmarks/mini/runs/overnight/mini_rank_0.log
#   docker ps --filter name=sweb      # running containers
#   df -h /                           # disk usage
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
NUM_GPUS="${NUM_GPUS:-8}"
GPUS_PER_SERVER="${GPUS_PER_SERVER:-8}"   # single TP8 replica
WORKERS="${WORKERS:-16}"                  # concurrent mini-swe-agent workers
PORT="${PORT:-4000}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/benchmarks/mini/runs/overnight}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DOCKER_PRUNE_INTERVAL="${DOCKER_PRUNE_INTERVAL:-120}"   # seconds between Docker image pruning

CONFIG="$PROJECT_ROOT/benchmarks/mini/swebench_vllm.yaml"
REGISTRY="$PROJECT_ROOT/benchmarks/mini/model_registry.json"
PYTHON="$(conda run -n swebench311 which python 2>/dev/null || which python3)"
MINI="$(conda run -n swebench311 which mini-extra 2>/dev/null || echo mini-extra)"
VLLM="$(conda run -n swebench311 which vllm 2>/dev/null || echo vllm)"

mkdir -p "$OUTPUT"

echo "======================================================================"
echo "  overnight SWE-bench run"
echo "======================================================================"
echo "  Model:       $MODEL"
echo "  GPUs:        $NUM_GPUS total, $GPUS_PER_SERVER/server"
echo "  Workers:     $WORKERS"
echo "  Output:      $OUTPUT"
echo "  Dataset:     $SUBSET/$SPLIT (all instances)"
echo "  Started at:  $(date)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 1. Disk pre-flight
# ---------------------------------------------------------------------------
AVAIL_GB=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
echo ""
echo "[pre-flight] Disk available on /: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 40 ]; then
    echo "[pre-flight] WARNING: less than 40 GB free — pruning stale Docker images first"
    docker image prune -f --filter "dangling=true" || true
    docker images --format "{{.Repository}}:{{.Tag}}" \
        | grep -E "swebench/sweb|sweagent/sweb" \
        | xargs -r docker rmi 2>/dev/null || true
    AVAIL_GB=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
    echo "[pre-flight] Disk available after prune: ${AVAIL_GB} GB"
fi
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "[pre-flight] FATAL: only ${AVAIL_GB} GB free — cannot safely run. Free space first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Start background Docker image pruner
#    Tries to remove swebench images NOT currently in use every N seconds.
#    docker rmi fails silently for images attached to running containers.
# ---------------------------------------------------------------------------
echo ""
echo "[pruner] Starting background Docker image pruner (interval: ${DOCKER_PRUNE_INTERVAL}s)"

docker_pruner() {
    while true; do
        sleep "$DOCKER_PRUNE_INTERVAL"
        # Dangling images (no tag, leftover build artifacts)
        docker image prune -f --filter "dangling=true" >/dev/null 2>&1 || true
        # SWE-bench eval images that are not currently backing a running container
        docker images --format "{{.Repository}}:{{.Tag}}" \
            | grep -E "swebench/sweb|sweagent/sweb" \
            | while IFS= read -r img; do
                docker rmi "$img" >/dev/null 2>&1 || true
              done
    done
}
docker_pruner &
PRUNER_PID=$!
echo "[pruner] PID: $PRUNER_PID"

# ---------------------------------------------------------------------------
# 3. Start vLLM server (single TP-8 replica, all 8 GPUs)
# ---------------------------------------------------------------------------
SERVER_LOG="$OUTPUT/server_rank_0.log"
echo ""
echo "[vllm] Starting vLLM: $MODEL  (TP=${GPUS_PER_SERVER}, port=${PORT})"
echo "[vllm] Log: $SERVER_LOG"

CUDA_VISIBLE_DEVICES=$(python3 -c "import os; n=$NUM_GPUS; print(','.join(str(i) for i in range(n)))")
export CUDA_VISIBLE_DEVICES

"$VLLM" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$GPUS_PER_SERVER" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype auto \
    --trust-remote-code \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    >"$SERVER_LOG" 2>&1 &
VLLM_PID=$!
echo "[vllm] PID: $VLLM_PID"

# ---------------------------------------------------------------------------
# 4. Wait for vLLM to be ready
# ---------------------------------------------------------------------------
TIMEOUT=900
DEADLINE=$((SECONDS + TIMEOUT))
echo "[vllm] Waiting up to ${TIMEOUT}s for port ${PORT} ..."
while true; do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); r=s.connect_ex(('127.0.0.1',$PORT)); s.close(); exit(0 if r==0 else 1)" 2>/dev/null; then
        echo "[vllm] ✓ Ready on port ${PORT}"
        break
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "[vllm] ✗ Did not respond within ${TIMEOUT}s — check $SERVER_LOG"
        kill "$PRUNER_PID" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[vllm] ✗ vLLM process died — check $SERVER_LOG"
        kill "$PRUNER_PID" 2>/dev/null || true
        exit 1
    fi
    ELAPSED=$((SECONDS - (DEADLINE - TIMEOUT)))
    printf "\r[vllm]   ... still loading (%ds/%ds)" "$ELAPSED" "$TIMEOUT"
    sleep 5
done

# ---------------------------------------------------------------------------
# 5. Run mini-swe-agent on all 500 instances (skips already-completed ones)
# ---------------------------------------------------------------------------
MINI_LOG="$OUTPUT/mini_rank_0.log"
REPLICA_OUT="$OUTPUT/replica_0"
mkdir -p "$REPLICA_OUT"

LITELLM_MODEL="hosted_vllm/$MODEL"
API_BASE="http://127.0.0.1:${PORT}/v1"

echo ""
echo "[mini] Starting mini-swe-agent workers"
echo "[mini]   Workers:    $WORKERS"
echo "[mini]   Model:      $LITELLM_MODEL"
echo "[mini]   API base:   $API_BASE"
echo "[mini]   Output:     $REPLICA_OUT"
echo "[mini]   Log:        $MINI_LOG"

LITELLM_MODEL_REGISTRY_PATH="$REGISTRY" \
MSWEA_COST_TRACKING="ignore_errors" \
"$MINI" swebench \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --output "$REPLICA_OUT" \
    --workers "$WORKERS" \
    -c "$CONFIG" \
    -c "model.model_name=$LITELLM_MODEL" \
    -c "model.model_kwargs.api_base=$API_BASE" \
    >"$MINI_LOG" 2>&1 &
MINI_PID=$!
echo "[mini] PID: $MINI_PID"

# ---------------------------------------------------------------------------
# 6. Monitor until mini finishes
# ---------------------------------------------------------------------------
echo ""
echo "[monitor] Watching progress (updates every 60s). Log: $MINI_LOG"
echo "[monitor] ctrl-C → graceful shutdown"

cleanup() {
    echo ""
    echo "[shutdown] Signal received — stopping workers ..."
    kill "$MINI_PID"  2>/dev/null || true
    kill "$VLLM_PID"  2>/dev/null || true
    kill "$PRUNER_PID" 2>/dev/null || true
    wait "$MINI_PID"  2>/dev/null || true
    wait "$VLLM_PID"  2>/dev/null || true
    echo "[shutdown] Done. Partial predictions at: $REPLICA_OUT/preds.json"
    exit 0
}
trap cleanup INT TERM

while kill -0 "$MINI_PID" 2>/dev/null; do
    DONE=$(python3 -c "
import json, pathlib
p = pathlib.Path('$REPLICA_OUT/preds.json')
print(len(json.loads(p.read_text())) if p.exists() else 0)
" 2>/dev/null || echo 0)
    DISK=$(df -h / | awk 'NR==2 {print $4}')
    IMG_COUNT=$(docker images --format "{{.Repository}}" | grep -cE "swebench/sweb|sweagent/sweb" 2>/dev/null || echo 0)
    echo "[monitor] $(date '+%H:%M:%S')  done=${DONE}/500  disk_free=${DISK}  sweb_images=${IMG_COUNT}"
    sleep 60
done

# ---------------------------------------------------------------------------
# 7. Collect exit code and finalize
# ---------------------------------------------------------------------------
wait "$MINI_PID"
MINI_RC=$?

echo ""
echo "[mini] Worker exited (rc=$MINI_RC)"

# Merge preds.json to top-level (in case caller expects it there)
if [ -f "$REPLICA_OUT/preds.json" ]; then
    cp "$REPLICA_OUT/preds.json" "$OUTPUT/preds.json"
    N=$(python3 -c "import json; d=json.load(open('$OUTPUT/preds.json')); print(len(d))")
    echo "[result] $N predictions → $OUTPUT/preds.json"
else
    echo "[result] WARNING: no preds.json found in $REPLICA_OUT"
fi

# ---------------------------------------------------------------------------
# 8. Shutdown
# ---------------------------------------------------------------------------
echo ""
echo "[shutdown] Terminating vLLM ..."
kill "$VLLM_PID"  2>/dev/null || true
wait "$VLLM_PID"  2>/dev/null || true
kill "$PRUNER_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Run complete — $(date)"
echo "======================================================================"
echo "  Predictions:  $OUTPUT/preds.json"
echo "  Trajectories: $REPLICA_OUT/"
echo "  vLLM log:     $SERVER_LOG"
echo "  mini log:     $MINI_LOG"
echo ""
echo "  Next: evaluate with sb-cli (fast, free cloud eval):"
echo "    sb-cli submit swe-bench_verified test \\"
echo "      --predictions_path $OUTPUT/preds.json \\"
echo "      --run_id qwen35_27b_$(date +%Y%m%d)"
echo ""
echo "  Or local eval:"
echo "    python -m swebench.harness.run_evaluation \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Verified \\"
echo "      --predictions_path $OUTPUT/preds.json \\"
echo "      --max_workers 4 --run_id qwen35_27b_$(date +%Y%m%d)"
echo "======================================================================"

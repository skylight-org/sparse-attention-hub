# SWE-bench with mini-swe-agent + vLLM (Qwen3.5-27B)

## Prerequisites

```bash
conda activate swebench311
pip install mini-swe-agent sb-cli   # if not already installed
docker ps                           # must work (Docker daemon running)
```

---

## Run all 500 instances

```bash
conda activate swebench311
python scripts/run_mini_swebench.py \
    --model Qwen/Qwen3.5-27B \
    --output benchmarks/mini/runs/my_run \
    --num-gpus 8 --gpus-per-server 8 \
    --workers 16
```

This starts vLLM on all 8 GPUs, runs all 500 SWE-bench Verified instances with 16 concurrent
workers, then writes `benchmarks/mini/runs/my_run/preds.json`. Takes ~5-8 hours.

If your SSH session drops, the run dies. To protect against that, wrap in `tmux`:

```bash
tmux new -s run500
# ... then run the command above inside tmux
# Detach: Ctrl-B D   Reattach: tmux attach -t run500
```

---

## Smoke test (1 instance)

```bash
conda activate swebench311
python scripts/run_mini_swebench.py \
    --model Qwen/Qwen3.5-27B \
    --output benchmarks/mini/runs/smoke \
    --num-gpus 8 --gpus-per-server 8 \
    --workers 1 \
    --filter '^django__django-11099$'
```

---

## Monitor a run

```bash
tail -f benchmarks/mini/runs/my_run/run.log          # overall progress
tail -f benchmarks/mini/runs/my_run/mini_rank_0.log  # agent details
```

---

## Evaluate results

```bash
# First time only:
sb-cli gen-api-key your@email.com
export SWEBENCH_API_KEY=<key from email>
sb-cli verify-api-key <code from email>

# Submit:
sb-cli submit swe-bench_verified test \
    --predictions_path benchmarks/mini/runs/my_run/preds.json \
    --run_id my_run_$(date +%Y%m%d)
```

---

## Key files

| File | What it does |
|------|-------------|
| `scripts/run_overnight.sh` | Full 500-instance overnight runner |
| `scripts/run_mini_swebench.py` | Flexible launcher (smoke tests, slices, multi-GPU) |
| `benchmarks/mini/swebench_vllm.yaml` | Agent config (prompts, sampling, thinking mode) |
| `benchmarks/mini/model_registry.json` | LiteLLM cost registry for local models |

---

## Disk cleanup (if running low on space)

```bash
# Remove only SWE-bench images:
docker images --format '{{.Repository}}:{{.Tag}}' | grep 'swebench/sweb' | xargs -r docker rmi

# Full Docker reset:
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm   $(docker ps -aq) 2>/dev/null || true
docker rmi  $(docker images -aq) 2>/dev/null || true
```

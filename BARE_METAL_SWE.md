# Bare-Metal SWE-bench Runs with Sparse Attention

This guide is for running the repo's sparse Hugging Face path on your own Linux GPU machine, not via Modal.

It covers:
- cloning and installing `sparse-attention-hub`
- authenticating with Hugging Face
- installing `mini-swe-agent` and local SWE-bench evaluation dependencies
- running `Qwen/Qwen3.5-27B` with the repo's current sparse config
- running specific SWE-bench Lite instances, including Django-only runs
- collecting sparse micro-metrics
- converting predictions into SWE-bench evaluation format
- evaluating with `python -m swebench.harness.run_evaluation`

This guide is written against the current local entrypoints in this repo:
- [`scripts/start_server.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/scripts/start_server.py)
- [`scripts/run_mini_swebench_hf_sparse.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/scripts/run_mini_swebench_hf_sparse.py)

## What This Path Does

The local sparse path is:
1. start an OpenAI-compatible local Hugging Face server
2. load `Qwen/Qwen3.5-27B`
3. use the sparse attention config defined in `scripts/start_server.py`
4. point `mini-swe-agent` at that server
5. run SWE-bench Lite instances in local Docker-backed environments
6. save both predictions and sparse micro-metrics to disk

The current sparse config in [`scripts/start_server.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/scripts/start_server.py) is:
- sink attention: `128`
- local window: `128`
- OracleTopK heavy size: `0.2`

In other words, this is the repo's current OracleTopK 20% setup.

## Recommended Hardware

Minimum guidance for this exact model path:
- Linux `x86_64`
- NVIDIA GPUs
- Docker installed and working
- enough aggregate VRAM to shard `Qwen/Qwen3.5-27B`

Safest starting point:
- `4 x 80GB` GPUs or better
- `workers=1`
- `--max-memory-per-gpu-gib 70`

Important:
- `mini-swe-agent`'s Docker SWE-bench path is intended for Linux x86 containers.
- Do not start with multiple batch workers on a 27B model unless you have already validated memory headroom.
- This path is optimized for correctness and stability first, not maximum throughput.

## Upstream References

Useful upstream docs:
- mini-SWE-agent SWE-bench usage: <https://mini-swe-agent.com/latest/usage/swebench/>
- mini-SWE-agent local model guide: <https://mini-swe-agent.com/latest/models/local_models/>
- mini-SWE-agent single-instance runner: <https://mini-swe-agent.com/latest/reference/run/swebench_single/>
- SWE-bench evaluation guide: <https://www.swebench.com/SWE-bench/guides/evaluation/>
- SWE-bench datasets guide: <https://www.swebench.com/SWE-bench/guides/datasets/>
- Qwen3.5-27B model card: <https://huggingface.co/Qwen/Qwen3.5-27B>

## 1. Clone the Repo

```bash
git clone https://github.com/xAlg-ai/sparse-attention-hub.git
cd sparse-attention-hub
```

If you are working from your existing checkout already, just `cd` into it.

## 2. Create or Activate a Python Environment

This repo already works well with Python `3.10`.

Example with conda:

```bash
conda create -n sparse-attention-hub python=3.10 -y
conda activate sparse-attention-hub
```

If you already have the environment, activate it:

```bash
conda activate sparse-attention-hub
```

## 3. Install Repo Dependencies

Install the repo itself first:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install the CUDA attention dependency required by the local HF server on GPU:

```bash
python -m pip install flash-attn
```

If `flash-attn` wheel resolution is tricky in your environment, the repo also defines an extra:

```bash
python -m pip install -e '.[flash_attn]'
```

## 4. Install Local SWE-bench / mini-SWE-agent Dependencies

These are not fully covered by the base repo install.

```bash
python -m pip install --upgrade mini-swe-agent swebench datasets
```

For `Qwen/Qwen3.5-27B`, the model card explicitly recommends a very recent `transformers`.
Install that next:

```bash
python -m pip install --upgrade 'transformers[serving] @ git+https://github.com/huggingface/transformers.git@main'
python -m pip install --upgrade accelerate sentencepiece torchvision pillow
```

## 5. Authenticate with Hugging Face

You need a Hugging Face token in the environment before large model download/load.

Recommended:

```bash
export HF_TOKEN=your_token_here
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login --token "$HF_TOKEN"
```

Notes:
- `Qwen/Qwen3.5-27B` is public, but a token still helps with download reliability and rate limits.
- If you use a shared machine, prefer environment variables plus `huggingface-cli login` under your own user.

## 6. Verify Docker Before You Start

The generation path and the evaluation path both rely on Docker.

```bash
docker info
```

If this fails, fix Docker first.

Important behavior:
- the first run for a repository can spend time building or warming Docker images
- that is normal
- do a one-instance smoke test before launching a large batch

## 7. Understand the Two Local Workflows

There are two ways to use the local sparse path.

### Option A: Recommended

Use the all-in-one helper:
- [`scripts/run_mini_swebench_hf_sparse.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/scripts/run_mini_swebench_hf_sparse.py)

This script:
- starts the sparse server for you
- runs `mini-extra swebench`
- writes `preds.json`
- writes `micro_metrics/micro_metrics.jsonl`
- writes `server.log` and `mini.log`
- stops the server when done

### Option B: Manual Debugging

Start [`scripts/start_server.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/scripts/start_server.py) yourself, test the API, then aim `mini-swe-agent` at it manually.

Use this only if you need finer debugging.

## 8. Start with a One-Instance Smoke Test

Do not start with all Django instances.
Start with one exact instance first.

Example using the helper script:

```bash
conda activate sparse-attention-hub
export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)

python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_smoke \
  --filter '^django__django-11001$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

What this does:
- starts `Qwen/Qwen3.5-27B` on GPUs `0,1,2,3`
- shards with `device_map=auto`
- caps each visible GPU at `70 GiB`
- runs exactly one SWE-bench Lite Django instance
- saves metrics and logs under `results/mini_swe/django_smoke`

## 9. Running All Django SWE-bench Lite Instances

Once the smoke test passes, run all Django Lite instances with a regex filter.

```bash
conda activate sparse-attention-hub
export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)

python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_lite_sparse \
  --filter '^django__django-[0-9]+$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

This is the simplest "all Django Lite" command in this repo today.

## 10. Running Specific Instance IDs

### Exact instance

```bash
python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_11001 \
  --filter '^django__django-11001$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

### First N matching instances

If you want a tiny batch before a full run, use `--slice` instead of `--filter`:

```bash
python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/lite_first_three \
  --slice '0:3' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

Important:
- `--filter` and `--slice` are mutually exclusive in the helper script.
- If you provide neither, the helper defaults to `0:5`.

## 11. Listing the Django IDs from SWE-bench Lite

If you want the exact Django instance IDs directly from the dataset instead of using the regex, run:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset('SWE-bench/SWE-bench_Lite', split='test')
django_ids = sorted(
    row['instance_id']
    for row in ds
    if row['instance_id'].startswith('django__django-')
)
print('\n'.join(django_ids))
print(f'\nTotal Django Lite instances: {len(django_ids)}')
PY
```

That gives you the exact instance list from the dataset you are about to run.

## 12. Manual Server Startup

If you want to launch only the sparse server and test it separately:

```bash
conda activate sparse-attention-hub
export SAH_METRICS_LOG_DIR=$PWD/results/server_only_metrics
mkdir -p "$SAH_METRICS_LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/start_server.py \
  --device-map auto \
  --max-memory-per-gpu-gib 70 \
  --is-hybrid \
  Qwen/Qwen3.5-27B \
  4000
```

Then, in another shell, sanity-check the endpoint:

```bash
curl http://127.0.0.1:4000/v1/models
```

Important:
- the helper script already starts and stops the server for you
- do not run the helper against a port that is already occupied by another copy of the server

## 13. Manual `mini-extra swebench-single` Debugging

If you want to debug one instance directly against a server you started yourself, use
`mini-extra swebench-single`.

The installed default config path in this environment is:

```text
/opt/anaconda3/envs/sparse-attention-hub/lib/python3.10/site-packages/minisweagent/config/benchmarks/swebench.yaml
```

Example:

```bash
conda activate sparse-attention-hub

mini-extra swebench-single \
  --subset lite \
  --split test \
  --instance django__django-11001 \
  --environment-class docker \
  --yolo \
  --exit-immediately \
  -c /opt/anaconda3/envs/sparse-attention-hub/lib/python3.10/site-packages/minisweagent/config/benchmarks/swebench.yaml \
  -c model.model_name=openai/Qwen/Qwen3.5-27B \
  -c model.model_kwargs.api_base=http://127.0.0.1:4000/v1 \
  -c model.model_kwargs.api_key=not-needed \
  -c model.model_kwargs.temperature=0.6 \
  -c model.model_kwargs.top_p=0.95 \
  -c model.model_kwargs.top_k=20 \
  -c model.model_kwargs.max_tokens=16384 \
  -o results/mini_swe/django_single_debug.traj.json
```

Why `--exit-immediately` matters:
- it prevents the agent from prompting for confirmation when it wants to finish
- that makes the single-instance flow safe for unattended or non-interactive runs

## 14. Where Outputs Go

For a run like:

```bash
python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_lite_sparse \
  --filter '^django__django-[0-9]+$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

expect:
- `results/mini_swe/django_lite_sparse/preds.json`
- `results/mini_swe/django_lite_sparse/micro_metrics/micro_metrics.jsonl`
- `results/mini_swe/django_lite_sparse/server.log`
- `results/mini_swe/django_lite_sparse/mini.log`

The helper script prints those paths again at the end.

## 15. Micro-Metrics

Sparse micro-metrics are dumped automatically by the server when `SAH_METRICS_LOG_DIR` is set.
The helper script sets that for you.

The file to look at is:

```text
<output>/micro_metrics/micro_metrics.jsonl
```

This is the main artifact if you want to analyze sparse behavior per request or per layer.

## 16. Prediction Format: `preds.json` vs SWE-bench JSONL

The local helper produces `preds.json`.
SWE-bench evaluation expects JSONL entries shaped like:

```json
{"instance_id": "repo__repo-123", "model_name_or_path": "your-model", "model_patch": "diff --git ..."}
```

Convert the helper output like this:

```bash
python - <<'PY'
import json
from pathlib import Path

input_path = Path('results/mini_swe/django_lite_sparse/preds.json')
output_path = input_path.with_suffix('.jsonl')
model_name = 'Qwen/Qwen3.5-27B-oracletopk20'

preds = json.loads(input_path.read_text())
with output_path.open('w', encoding='utf-8') as f:
    for instance_id, patch in preds.items():
        row = {
            'instance_id': instance_id,
            'model_name_or_path': model_name,
            'model_patch': patch,
        }
        f.write(json.dumps(row) + '\n')

print(output_path)
PY
```

This creates:

```text
results/mini_swe/django_lite_sparse/preds.jsonl
```

## 17. Evaluate with SWE-bench

Once you have `preds.jsonl`, evaluate locally.

### Evaluate the predictions file directly

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path results/mini_swe/django_lite_sparse/preds.jsonl \
  --max_workers 1 \
  --cache_level env \
  --clean True \
  --run_id django_lite_sparse_eval
```

### Evaluate only a few specific instances

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path results/mini_swe/django_lite_sparse/preds.jsonl \
  --instance_ids django__django-11001 django__django-11039 \
  --max_workers 1 \
  --cache_level env \
  --clean True \
  --run_id django_subset_eval
```

### What the conservative evaluation flags do

- `--max_workers 1`: minimizes local CPU/RAM and Docker pressure
- `--cache_level env`: reuses environment-level images but avoids the heaviest cache growth
- `--clean True`: cleans intermediate resources automatically after evaluation

Those are slower than an aggressive setup, but much less likely to blow up disk or memory.

## 18. Avoiding OOMs

Use these defaults unless you have already validated a more aggressive setup:

### Generation-side safety

- keep `--workers 1`
- keep `--visible-gpus` fixed and explicit
- keep `--max-memory-per-gpu-gib` below full card capacity
- start with one exact instance
- do not run multiple helper-script jobs against the same GPUs

### Recommended first settings on 80GB cards

- `--visible-gpus 0,1,2,3`
- `--max-memory-per-gpu-gib 70`
- `--workers 1`

### If you still OOM during model load

- increase number of GPUs used for sharding
- lower `--max-memory-per-gpu-gib` only if you need more headroom against fragmentation, not as a first fix
- make sure no other jobs are using those GPUs
- verify `transformers` is up to date

### If you OOM later during agent trajectories

- keep `workers=1`
- reduce concurrent activity on the box
- test on smaller slices first
- consider a smaller model if your hardware is marginal

## 19. Docker Build Behavior

You asked specifically about building Docker images for sample runs.

The practical answer for this repo is:
- you do not need a separate custom image-build step before using `scripts/run_mini_swebench_hf_sparse.py`
- the required SWE-bench environment images are built or pulled on demand during the run
- the first instance for a repo is usually the slowest because of this

Best practice:
- run a smoke instance first for each repo family you care about
- keep Docker cache warm on the machine
- do not prune Docker in the middle of a benchmark campaign

## 20. Common Failures

### `mini-extra: command not found`

Your conda env is not active, or `mini-swe-agent` is not installed there.

Fix:

```bash
conda activate sparse-attention-hub
python -m pip install --upgrade mini-swe-agent
```

### `Docker is required and does not appear to be running`

The helper script explicitly checks `docker info`.
Start Docker and retry.

### `Server did not become ready`

Read:
- `<output>/server.log`

Common causes:
- missing `flash-attn`
- old `transformers`
- not enough VRAM
- another process already using the chosen GPUs

### Qwen model load fails because the architecture is unsupported

The Qwen model card says the latest `transformers` is required.
Upgrade again:

```bash
python -m pip install --upgrade 'transformers[serving] @ git+https://github.com/huggingface/transformers.git@main'
```

Then retry.

### Sparse metrics file is missing

Check:
- `<output>/server.log`

The helper script warns if `micro_metrics.jsonl` was not created.
That usually means the server never fully served sparse requests.

## 21. Recommended End-to-End Order

If you just want the safest sequence:

1. Activate env and install everything in this guide.
2. Log into Hugging Face.
3. Verify `docker info`.
4. Run a one-instance Django smoke test.
5. Inspect `server.log`, `mini.log`, and `micro_metrics.jsonl`.
6. Run all Django Lite with the regex filter.
7. Convert `preds.json` to `preds.jsonl`.
8. Run `python -m swebench.harness.run_evaluation` with `--max_workers 1`.

## 22. Most Important Commands

### Smoke test

```bash
python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_smoke \
  --filter '^django__django-11001$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

### All Django Lite

```bash
python scripts/run_mini_swebench_hf_sparse.py \
  --output results/mini_swe/django_lite_sparse \
  --filter '^django__django-[0-9]+$' \
  --workers 1 \
  --visible-gpus 0,1,2,3 \
  --max-memory-per-gpu-gib 70
```

### Evaluation

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path results/mini_swe/django_lite_sparse/preds.jsonl \
  --max_workers 1 \
  --cache_level env \
  --clean True \
  --run_id django_lite_sparse_eval
```

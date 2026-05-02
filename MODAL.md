# Modal Sparse SWE Runs

This document explains the working Modal path in this repo for:

- `Qwen/Qwen3.5-27B`
- Hugging Face backend
- sparse attention via `scripts/start_server.py`
- `mini-swe-agent` SWE-bench sample runs

This is the path that avoids local Docker and runs everything on Modal GPUs.

## What To Use

Main runner:
- [`benchmark/scripts/modal_mini_swe_local_sparse.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/benchmark/scripts/modal_mini_swe_local_sparse.py)

Patch extractor:
- [`benchmark/scripts/modal_extract_preds.py`](/Users/prithvidixit/Desktop/sky/sparse-attention-hub/benchmark/scripts/modal_extract_preds.py)

The runner:
- starts the sparse HF server inside Modal
- loads `Qwen/Qwen3.5-27B`
- runs `mini-extra swebench-single` against that local server
- writes per-instance trajectories and logs into a Modal volume
- writes `micro_metrics.jsonl` continuously into the same run directory

## One-Time Setup

From the conda env you have already been using:

```bash
modal setup
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
```

Modal volumes used by this flow:
- `mini-swe-sparse-results`
- `mini-swe-sparse-hf-cache`

`mini-swe-sparse-hf-cache` keeps model downloads across runs so retries do not redownload the full checkpoint every time.

## Important Behavior

This runner is sequential inside one GPU-backed Modal function:
- one model server
- one SWE instance at a time

That is deliberate:
- it avoids OOM risk on `Qwen3.5-27B`
- it keeps GPU usage predictable
- it saves progress after each instance

The current GPU setting in the script is:
- `H100:4`

## What Gets Saved

For a run id like `pylint_local_sparse_v8`, results are written under:

```text
/results/pylint_local_sparse_v8/
```

Inside that directory:
- `server.log`
- `micro_metrics/micro_metrics.jsonl`
- one directory per instance id

Inside each instance directory:
- `<instance_id>.traj.json`
- `minisweagent.log`

The runner commits the Modal results volume repeatedly:
- during server startup polling
- after each instance
- on shutdown

So if credits run out or the app is stopped, partial outputs already written to disk should still be in the volume.

## Run Pylint Samples

Command:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run \
  benchmark/scripts/modal_mini_swe_local_sparse.py \
  --filter '^pylint-dev__pylint-[0-9]+$' \
  --run-id pylint_local_sparse_v8
```

This matches the 6 `pylint-dev__pylint-*` lite test instances.

## Run Pytest Samples

Command:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run \
  benchmark/scripts/modal_mini_swe_local_sparse.py \
  --filter '^pytest-dev__pytest-[0-9]+$' \
  --run-id pytest_local_sparse_v1
```

This is the regex for the 17 `pytest-dev__pytest-*` lite instances.

If you want a smoke test first:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run \
  benchmark/scripts/modal_mini_swe_local_sparse.py \
  --filter '^pytest-dev__pytest-[0-9]+$' \
  --limit 1 \
  --run-id pytest_local_sparse_smoke
```

## Detached Runs

If you want the app to continue after disconnect:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run --detach \
  benchmark/scripts/modal_mini_swe_local_sparse.py \
  --filter '^pytest-dev__pytest-[0-9]+$' \
  --run-id pytest_local_sparse_v1
```

Then monitor it with:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal app logs <APP_ID> --tail 200 --timestamps
```

Stop it with:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal app stop <APP_ID> -y
```

## Extract Patches Into `preds.json`

Some runs can finish with:
- patch text saved in trajectory messages
- empty `info.submission`

That is why the extractor exists.

For pylint:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run \
  benchmark/scripts/modal_extract_preds.py \
  --run-id pylint_local_sparse_v8 \
  --instance-filter '^pylint-dev__pylint-[0-9]+$' \
  --output-name preds_pylint_samples.json \
  --summary-name preds_pylint_samples.summary.json
```

For pytest:

```bash
/opt/anaconda3/envs/sparse-attention-hub/bin/modal run \
  benchmark/scripts/modal_extract_preds.py \
  --run-id pytest_local_sparse_v1 \
  --instance-filter '^pytest-dev__pytest-[0-9]+$' \
  --output-name preds_pytest_samples.json \
  --summary-name preds_pytest_samples.summary.json
```

Outputs are written back into the run directory, for example:

```text
/results/pylint_local_sparse_v8/preds_pylint_samples.json
/results/pylint_local_sparse_v8/preds_pylint_samples.summary.json
```

## Current Known Limitation

The current runner uses the local-environment `mini-swe-agent` path rather than the original Docker SWE-bench path.

That means:
- the sparse model path is working
- the SWE trajectories and sparse metrics are saved
- patch recovery currently relies on saved trajectory content when the run exits without a populated `submission`

Use the extractor script to turn those saved trajectories into a `preds.json`-style file.

## Quick Checklist

1. Create the HF secret once.
2. Run `modal_mini_swe_local_sparse.py` with the instance regex you want.
3. Watch `modal app logs`.
4. If needed, stop the app with `modal app stop`.
5. Run `modal_extract_preds.py` after the run.
6. Collect:
   - `micro_metrics.jsonl`
   - trajectories
   - `preds_*.json`

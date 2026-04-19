# SWE-Bench inference with sparse-attention-hub

This guide walks through cloning dependencies, building Docker images, pointing OpenHands at your local Hugging Face server, and running **distributed SWE-Bench inference**. It ends with a **fully dense** Qwen3.5-27B example (no custom sparse attention).

---

## 1. Prerequisites

- **GPU machine** with NVIDIA driver and PyTorch that match .
- **Docker** (for SWE-Bench evaluation containers).
- **`uv`** (required by `scripts/distributed_swebench_inference.py` for environment validation).
- **`flash_attn`** if you run `scripts/start_server.py` on CUDA (see that script’s docstring).

Install this repo in the same Python you use for servers and the driver script:

```bash
cd sparse-attention-hub
pip install -e .
# or: uv sync && uv pip install -e .
```

For multi-GPU servers, child processes should use the same env as your GPUs:

```bash
conda activate "env"
export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)
```

---

## 2. Clone repositories

**sparse-attention-hub** (branch api-2 for now):

```bash
git clone https://github.com/skylight-org/sparse-attention-hub.git #ensure you clone api-2
cd sparse-attention-hub
```

**OpenHands benchmarks** (for SWE-Bench image build and harness):

```bash
git clone https://github.com/OpenHands/benchmarks
```

Follow upstream docs: [OpenHands/benchmarks](https://github.com/OpenHands/benchmarks). Typical steps are `make build` and building SWE-Bench images. One approach (from their tree):

#make sure you run make build first

```bash
cd benchmarks
uv run python -m benchmarks.swebench.build_images \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push \
  --max-workers 32
```

Adjust flags to match your registry and whether you push images.

---

## 3. Dense vs sparse (`start_server.py`)

The HF server reads **`SPARSE_CONFIG`** at the top of `scripts/start_server.py`.

- **Dense (standard Hugging Face attention):** set **`SPARSE_CONFIG = None`** (and remove or comment out the `ResearchAttentionConfig(...)` block if you are editing manually).
- **Sparse (research attention):** keep a `ResearchAttentionConfig` with maskers (sink, local, OracleTopK, etc.).

For the **Qwen3.5-27B dense** command below, ensure **`SPARSE_CONFIG = None`** before starting servers.

---

## 4. LLM config file (OpenHands / LiteLLM)

`scripts/distributed_swebench_inference.py` **generates** a temporary JSON per replica via `create_llm_config()`. You do not have to create this by hand unless you run the harness yourself. The content is equivalent to:

```json
{
  "model": "openai/Qwen/Qwen3.5-27B",
  "base_url": "http://172.17.0.1:4000/v1",
  "api_key": "not-needed",
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_output_tokens": 32768,
  "litellm_extra_body": {
    "repetition_penalty": 1.0,
    "min_p": 0.0,
    "presence_penalty": 0.0
  }
}
```

- **`model`:** LiteLLM id; often `openai/<HF model id>` when using an OpenAI-compatible endpoint.
- **`base_url`:** Must be reachable **from inside Docker**, not only from the host. See §5.
- **`api_key`:** Placeholder; local server ignores it if unused.
- **`temperature` / `top_p` / `top_k` / `max_output_tokens`:** Sampling and length caps passed to the server.
- **`litellm_extra_body`:** Extra keys forwarded to LiteLLM; local `start_server` uses repetition-related fields where implemented.

Replica **1** uses port **`4001`** in `base_url` when `--base_port` is `4000` (i.e. `http://172.17.0.1:4001/v1`).

---

## 5. Docker: routing the LLM host

SWE-Bench runs **inside containers**. On the default Linux bridge, **`localhost` inside a container is not the host**. The driver script builds URLs like `http://<host>:<port>/v1` for each replica.

**Default behavior** (`scripts/distributed_swebench_inference.py`):

1. If **`SWEBENCH_DISTRIBUTED_LLM_HOST`** is set, use it (hostname or IP, no `http://` required).
2. Else try the **Docker bridge gateway** from `docker network inspect bridge` (often **`172.17.0.1`**).
3. Else fall back to **`172.17.0.1`**.

**Override when needed:**

```bash
export SWEBENCH_DISTRIBUTED_LLM_HOST=your hosts LAN IP
# or your host’s LAN IP if containers use a different network
```

Ensure the chosen host and ports (**`base_port`, `base_port+1`, …**) are allowed through any firewall between containers and the host.

---

## 6. Qwen3.5-27B — **dense** distributed SWE-Bench (example)

**Before running:** set **`SPARSE_CONFIG = None`** in `scripts/start_server.py` (§3).

**One shell:**

```bash
conda activate "env"
cd /path/to/sparse-attention-hub
export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)

python scripts/distributed_swebench_inference.py \
  --instances_file benchmarks/neurips/instances_qwen3.5-27b_32.txt \
  --model_name openai/Qwen/Qwen3.5-27B \
  --output_dir benchmarks/neurips/run/qwen35-27b-dense-4plus4 \
  --num_gpus 8 \
  --gpus-per-server 4 \
  --max-memory-per-gpu-gib 72 \
  --base_port 4000 \
  --num-workers 1
```

### Flag reference

| Flag | Meaning |
|------|--------|
| `--instances_file` | Text file: one SWE-Bench instance id per line (see §7). |
| `--model_name` | LiteLLM model id passed to OpenHands; must match what the server exposes. Provider prefix `openai/` is stripped for **starting** the HF server; the full string is used in the LLM config for the client. |
| `--output_dir` | Run artifacts: server logs (`server_rank_*.log`), per-GPU harness output, merged reports. |
| `--num_gpus` | Total physical GPUs used on this machine for this run (after `--gpu_offset`). |
| `--gpus-per-server` | GPUs per **`start_server.py` process** (`CUDA_VISIBLE_DEVICES` length). Must divide `--num_gpus`. `4` on an 8-GPU box ⇒ **two replicas** on ports `4000` and `4001`. |
| `--max-memory-per-gpu-gib` | Passed to HF **`max_memory`** when using `--device-map auto` (headroom on 80GB cards, e.g. `72`). |
| `--base_port` | First replica port; replica *k* uses `base_port + k`. |
| `--num-workers` | Parallel SWE-Bench instances per **GPU track** (OpenHands). Use **`1`** for large models to reduce concurrent load on each server. |

**Omitted flags (defaults):** `--gpu_offset` defaults to `0`. **`--is-hybrid`** is omitted for dense. **`--skip_validation`** is omitted so `uv`/docker checks run.

Optional env: **`SWEBENCH_SERVER_READY_TIMEOUT`** (seconds, default `900`) while waiting for each server to load.

---

## 7. Instance list — 32× scikit-learn (SWE-bench Verified)

These match `benchmarks/neurips/instances_qwen3.5-27b_32.txt`:

```
scikit-learn__scikit-learn-10297
scikit-learn__scikit-learn-10844
scikit-learn__scikit-learn-10908
scikit-learn__scikit-learn-11310
scikit-learn__scikit-learn-11578
scikit-learn__scikit-learn-12585
scikit-learn__scikit-learn-12682
scikit-learn__scikit-learn-12973
scikit-learn__scikit-learn-13124
scikit-learn__scikit-learn-13135
scikit-learn__scikit-learn-13142
scikit-learn__scikit-learn-13328
scikit-learn__scikit-learn-13439
scikit-learn__scikit-learn-13496
scikit-learn__scikit-learn-13779
scikit-learn__scikit-learn-14053
scikit-learn__scikit-learn-14087
scikit-learn__scikit-learn-14141
scikit-learn__scikit-learn-14496
scikit-learn__scikit-learn-14629
scikit-learn__scikit-learn-14710
scikit-learn__scikit-learn-14894
scikit-learn__scikit-learn-14983
scikit-learn__scikit-learn-15100
scikit-learn__scikit-learn-25102
scikit-learn__scikit-learn-25232
scikit-learn__scikit-learn-25747
scikit-learn__scikit-learn-25931
scikit-learn__scikit-learn-25973
scikit-learn__scikit-learn-26194
scikit-learn__scikit-learn-26323
scikit-learn__scikit-learn-9288
```

---

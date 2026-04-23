# SWE-Bench inference with sparse-attention-hub

This guide walks through cloning dependencies, building Docker images, and running **distributed SWE-Bench inference** with a single launcher script that can start either:

- a Hugging Face adapter server (`--backend hf`), or
- a vLLM OpenAI-compatible server (`--backend vllm`).

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

## 3. Backend switching (command-only)

`scripts/distributed_swebench_inference.py` now supports:

- **`--backend hf`**: launch `scripts/start_server.py` replicas (sparse/dense behavior from that server config).
- **`--backend vllm`**: launch `vllm serve` replicas directly.

So switching HF vs vLLM does **not** require editing `distributed_swebench_inference.py`; it is a command flag.

Notes:

- If you use `--backend hf`, sparse vs dense still depends on `SPARSE_CONFIG` in `scripts/start_server.py`.
- If you use `--backend vllm`, `SPARSE_CONFIG` is irrelevant.

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
  "max_output_tokens": 8192,
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

## 6. Qwen3.5-27B — distributed SWE-Bench (HF or vLLM)

### 6.1 HF backend (`--backend hf`)

If you want dense HF behavior, set `SPARSE_CONFIG = None` in `scripts/start_server.py`.

```bash
conda activate "env"
cd /path/to/sparse-attention-hub
export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)

python scripts/distributed_swebench_inference.py \
  --backend hf \
  --instances_file benchmarks/neurips/instances_qwen3.5-27b_32.txt \
  --model_name openai/Qwen/Qwen3.5-27B \
  --output_dir benchmarks/neurips/run/qwen35-27b-hf-dense-4plus4 \
  --num_gpus 8 \
  --gpus-per-server 4 \
  --max-memory-per-gpu-gib 72 \
  --base_port 4000 \
  --num-workers 1
```

### 6.2 vLLM backend (`--backend vllm`)

```bash
conda activate "env"
cd /path/to/sparse-attention-hub

python scripts/distributed_swebench_inference.py \
  --backend vllm \
  --instances_file benchmarks/neurips/instances_qwen3.5-27b_32.txt \
  --model_name openai/Qwen/Qwen3.5-27B \
  --output_dir benchmarks/neurips/run/qwen35-27b-vllm-4plus4 \
  --num_gpus 8 \
  --gpus-per-server 4 \
  --base_port 4000 \
  --num-workers 1 \
  --vllm-max-model-len 131072 \
  --vllm-gpu-memory-utilization 0.92 \
  --vllm-dtype auto
```

### Flag reference

| Flag | Meaning |
|------|--------|
| `--instances_file` | Text file: one SWE-Bench instance id per line (see §7). |
| `--model_name` | LiteLLM model id passed to OpenHands; must match what the server exposes. Provider prefix `openai/` is stripped for **starting** the HF server; the full string is used in the LLM config for the client. |
| `--backend` | Server backend: `hf` or `vllm`. |
| `--output_dir` | Run artifacts: server logs (`server_rank_*.log`), per-GPU harness output, merged reports. |
| `--num_gpus` | Total physical GPUs used on this machine for this run (after `--gpu_offset`). |
| `--gpus-per-server` | GPUs per server process (HF or vLLM). Must divide `--num_gpus`. `4` on an 8-GPU box ⇒ **two replicas** on ports `4000` and `4001`. |
| `--max-memory-per-gpu-gib` | HF only: passed to `start_server.py` `max_memory` with `device_map=auto` (e.g. `72`). |
| `--base_port` | First replica port; replica *k* uses `base_port + k`. |
| `--num-workers` | Parallel SWE-Bench instances per **GPU track** (OpenHands). Use **`1`** for large models to reduce concurrent load on each server. |
| `--vllm-max-model-len` | vLLM only: server context window cap. |
| `--vllm-gpu-memory-utilization` | vLLM only: memory fraction per GPU. |
| `--vllm-dtype` | vLLM only: dtype (e.g., `auto`, `bfloat16`). |

**Omitted flags (defaults):** `--gpu_offset` defaults to `0`, `--skip_validation` omitted so `uv`/docker checks run. `--is-hybrid` is HF-only (forwarded to `start_server.py`).

Optional env: **`SWEBENCH_SERVER_READY_TIMEOUT`** (seconds, default `900`) while waiting for each server to load.

### vLLM tuning notes (`--backend vllm`)

Align with the **[Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)** (vLLM section):

- **Long context:** The model is trained for **262,144** tokens. A low **`--max-model-len` (e.g. 32k)** forces tiny `max_output_tokens` and frequent `VLLMValidationError`. Use the **largest `--max-model-len` that fits in VRAM** (try **131072** or **262144** on 8×80GB; reduce if you OOM).
- **Tool calls:** Official recipe uses **`--reasoning-parser qwen3`**, **`--enable-auto-tool-choice`**, and **`--tool-call-parser qwen3_coder`** (not `qwen3_xml`).
- **SWE-Bench is text-only:** **`--language-model-only`** skips the vision encoder and frees memory for **more KV / longer context** (recommended for this harness).
- **Direct (non-thinking) replies:** Qwen3.5 “thinks” by default; for coding agents, pass **`chat_template_kwargs: {"enable_thinking": false}`** (already in `benchmarks/neurips/llm_vllm_dense.json` via `litellm_extra_body`).
- **Sampling:** The card recommends **temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0** for *precise coding* in thinking mode; the same JSON uses that profile for instruct-style runs.

Equivalent standalone server (if you are not using the distributed launcher):

```bash
conda activate swebench311
cd /path/to/sparse-attention-hub

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 4000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --dtype auto \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**Client:** Keep **`max_output_tokens` + prompt tokens ≤ `--max-model-len`**. With a **large** server cap, raise **`max_output_tokens`** in the LLM JSON toward the card’s **32k** typical / **~81k** heavy recommendation as VRAM allows. For a **32k** server only, keep **`8192`** (or lower) so prompts have headroom.

**Ultra-long (YaRN):** The card documents **`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`** and **`--hf-overrides`** for **1M+** context; use only if you need beyond native 262k and know the tradeoffs.

**Stopping vLLM:** Use **one** `Ctrl+C` and wait for “Application shutdown complete.” Hitting `Ctrl+C` repeatedly while requests are in flight can produce `cannot schedule new futures after shutdown`, `Worker proc … died unexpectedly`, and `RuntimeError: cancelled` in the logs—these are **teardown races**, not model errors. Stop `swebench-infer` first (or wait for idle), then stop vLLM. If a worker is stuck: `pkill -f 'vllm serve'` after a short grace period, then restart.

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

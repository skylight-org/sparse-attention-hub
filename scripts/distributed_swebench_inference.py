#!/usr/bin/env python3
"""
Distributed SWE-Bench Inference Script

This script distributes SWE-Bench inference across multiple GPUs on a single node.
It splits instances, starts multiple model-server replicas (HF or vLLM), and runs
parallel SWE-Bench inference tracks against those replicas.

Usage:
    python scripts/distributed_swebench_inference.py \
        --instances_file path/to/instances.txt \
        --model_name openai/Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --output_dir /path/to/output \
        --num_gpus 8 \
        --base_port 4000

    Switch backend with command-only changes:

        # HuggingFace adapter server (sparse/dense controlled by start_server.py)
        python scripts/distributed_swebench_inference.py ... --backend hf

        # vLLM OpenAI server
        python scripts/distributed_swebench_inference.py ... --backend vllm

    Qwen3.5-27B (hybrid) + sparse OracleTopK 20% on 8 GPUs (4+4 replicas), one SWE worker each::

        python scripts/distributed_swebench_inference.py \\
            --instances_file benchmarks/neurips/instances_qwen3.5-27b_32.txt \\
            --model_name openai/Qwen/Qwen3.5-27B \\
            --output_dir benchmarks/neurips/run/qwen35-27b-sparse-hybrid \\
            --num_gpus 8 \\
            --gpus-per-server 4 \\
            --max-memory-per-gpu-gib 72 \\
            --base_port 4000 \\
            --num-workers 1 \\
            --is-hybrid

    Per-replica MicroMetricLogger output is written under
    ``<output_dir>/metrics/replica_<k>/micro_metrics.jsonl`` (``SAH_METRICS_LOG_DIR``).

    Eight-instance smoke: use ``benchmarks/neurips/instances_8gpu_sparse_smoke.txt``
    with ``--num_gpus 8 --num-workers 1``. Sparse maskers vs dense are controlled by
    ``SPARSE_CONFIG`` in ``scripts/start_server.py`` (``None`` = fully dense).

    Multi-GPU per model (e.g. 27B on 4×H100, two replicas on 8×H100): set
    ``--gpus-per-server 4`` (must divide ``--num_gpus``). Servers are launched
    with ``CUDA_VISIBLE_DEVICES`` listing each shard and ``start_server.py
    --device-map auto`` (optional ``--max-memory-per-gpu-gib``).

    Large models (tight VRAM): pass ``--num-workers 1`` so each GPU process runs
    one instance at a time. Conversation and LiteLLM timeouts default to 5400s
    (90 minutes) unless overridden in the environment. Per-completion OpenHands
    ``LLM`` HTTP ``timeout`` in the generated JSON defaults to the same value;
    override with ``SWEBENCH_LLM_HTTP_TIMEOUT`` (seconds).

The LLM host for Docker workspaces is chosen automatically: the default ``bridge``
network gateway from ``docker network inspect`` (when Docker is available), otherwise
``172.17.0.1``. Override only if needed via env ``SWEBENCH_DISTRIBUTED_LLM_HOST``.

Model servers can take a long time to become ready; the default wait is 900 seconds,
overridable with env ``SWEBENCH_SERVER_READY_TIMEOUT``.

Child ``start_server.py`` processes use the same interpreter as this script
(``sys.executable``) unless you override it. Prefer a **GPU-ready** env whose
PyTorch matches your NVIDIA driver (often **conda ``swebench311``** on this project).

Set **one** of:

- ``export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)`` after
  ``conda activate swebench311`` (recommended for multi-GPU ``--device-map auto``).
- ``export SWEBENCH_SERVER_PYTHON=...`` (alias for the same override).

If the server log shows ``--device-map auto requires CUDA`` or a driver/CUDA mismatch
warning, the server Python is almost always wrong: the parent script may be
``uv run``/``.venv`` while GPUs need your conda env.

Requires ``uvicorn``, ``fastapi``, ``sparse_attention_hub``, and optionally
``flash_attn`` (FlashAttention-2) in that interpreter.
"""

#pd (things to look at later if we want to optimize)
#i think we need to print the config in the terminal when we launch this, rn need to do some tail command business
#look into slurm? but depending on optimization this could be ok
# look into max retries flag
#memmory requirements for this? on 80gb h100 am running close to memory

#tested command:
# python scripts/distributed_swebench_inference.py   --instances_file scripts/example_instances.txt   --model_name openai/Qwen/Qwen3-Coder-30B-A3B-Instruct   --output_dir distributed_results/   --num_gpus 2   --base_port 4000


import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import socket


def _normalize_llm_host(raw: str) -> str:
    """Strip scheme/path from a host string for use in ``http://HOST:PORT/v1``."""
    candidate: str = raw.strip()
    lower: str = candidate.lower()
    if lower.startswith("http://"):
        candidate = candidate[7:]
    elif lower.startswith("https://"):
        candidate = candidate[8:]
    return candidate.split("/")[0]


def _detect_docker_bridge_gateway() -> Optional[str]:
    """Return the IPv4 gateway of Docker's default ``bridge`` network, if discoverable.

    Containers on the default bridge reach the host via this gateway. Falls back to
    None if Docker is unavailable or the gateway cannot be read.

    Returns:
        Gateway IP string, or None.
    """
    try:
        proc: subprocess.CompletedProcess[str] = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "-f",
                "{{(index .IPAM.Config 0).Gateway}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    gw: str = proc.stdout.strip()
    if not gw or gw == "<no value>":
        return None
    return gw


def resolve_server_python() -> str:
    """Python executable used for each ``start_server.py`` subprocess.

    Resolution order:

    1. ``SPARSE_ATTENTION_SERVER_PYTHON`` if set and non-empty.
    2. ``SWEBENCH_SERVER_PYTHON`` if set and non-empty (alias).
    3. ``sys.executable`` (the interpreter running this script).

    Returns:
        Path or command string passed to ``subprocess.Popen`` as the server binary.
    """
    for key in ("SPARSE_ATTENTION_SERVER_PYTHON", "SWEBENCH_SERVER_PYTHON"):
        candidate: str = os.environ.get(key, "").strip()
        if candidate:
            return candidate
    return sys.executable


def resolve_vllm_bin() -> str:
    """vLLM executable for server launch (defaults to ``vllm``)."""
    return os.environ.get("SWEBENCH_VLLM_BIN", "vllm").strip() or "vllm"


def strip_provider_prefix(model_name: str) -> str:
    """Drop LiteLLM/OpenAI-style provider prefix for local server startup."""
    if "/" not in model_name:
        return model_name
    parts: List[str] = model_name.split("/")
    if parts[0] in {"openai", "anthropic", "google", "huggingface", "litellm"}:
        return "/".join(parts[1:])
    return model_name


def resolve_llm_host() -> str:
    """Resolve hostname or IP for LLM ``base_url`` as seen from Docker workspaces.

    Docker containers cannot use ``localhost`` to reach servers on the host. Order:
    ``SWEBENCH_DISTRIBUTED_LLM_HOST`` if set; else gateway from ``docker network inspect
    bridge``; else ``172.17.0.1``.

    Returns:
        Host string without URL scheme or path (ports are added per-GPU by the caller).
    """
    env_host: str = os.environ.get("SWEBENCH_DISTRIBUTED_LLM_HOST", "").strip()
    if env_host:
        return _normalize_llm_host(env_host)
    bridge_gw: Optional[str] = _detect_docker_bridge_gateway()
    if bridge_gw:
        return _normalize_llm_host(bridge_gw)
    return "172.17.0.1"


def split_instances_file(instances_file: str, num_splits: int) -> List[str]:
    """Split instances file into num_splits parts."""
    with open(instances_file, 'r') as f:
        instances = [line.strip() for line in f if line.strip()]

    if not instances:
        raise ValueError(f"No instances found in {instances_file}")

    #calculate nu of instances per split
    instances_per_split = len(instances) // num_splits
    remainder = len(instances) % num_splits

    split_files = []
    start_idx = 0

    for i in range(num_splits):
        #split the remainder across first few splits (ideally should just be split evenly tho)
        split_size = instances_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + split_size

        split_instances = instances[start_idx:end_idx]

        # temporary file for this split (for each gpu)
        split_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        split_file.write('\n'.join(split_instances) + '\n')
        split_file.close()

        split_files.append(split_file.name)
        start_idx = end_idx

    print(f"Split {len(instances)} instances into {num_splits} files:")
    for i, split_file in enumerate(split_files):
        with open(split_file, 'r') as f:
            count = len([line for line in f if line.strip()])
        print(f"  GPU {i}: {count} instances")
    #ret split
    return split_files


def _llm_http_timeout_seconds() -> int:
    """Return OpenHands ``LLM`` per-request HTTP timeout in seconds.

    Long sparse / hybrid forwards can exceed the SDK default (300s). This value
    is written into the temp JSON as ``timeout`` so httpx/LiteLLM wait long enough.

    Returns:
        Parsed ``SWEBENCH_LLM_HTTP_TIMEOUT`` if set and valid, else ``5400``.
        Clamped to at least ``60``.

    Note:
        Align with ``CONVERSATION_TIMEOUT`` / ``LITELLM_TIMEOUT`` for SWE runs.
    """
    raw: str = os.environ.get("SWEBENCH_LLM_HTTP_TIMEOUT", "5400").strip()
    try:
        parsed: int = int(raw)
    except ValueError:
        return 5400
    return max(60, parsed)


def create_llm_config(base_url: str, model_name: str) -> str:
    """Create a temporary OpenHands LLM JSON config for a per-GPU server.

    Uses a "precise coding / thinking-style" sampling preset: slightly lower
    temperature, higher ``top_p``, and neutral repetition / penalties. Extra
    sampling keys that are not first-class OpenHands ``LLM`` fields are passed
    via ``litellm_extra_body`` (forwarded as LiteLLM ``extra_body``) for
    backends that honor them (e.g. vLLM); the local ``start_server.py`` HF path
    only applies temperature, ``top_p``, ``top_k``, and ``repetition_penalty``.

    Args:
        base_url: OpenAI-compatible API base (e.g. ``http://172.17.0.1:4000/v1``).
        model_name: LiteLLM model id (e.g. ``openai/Qwen/Qwen3.5-9B``).

    Returns:
        Path to the written JSON config file (caller should delete when done).

    Note:
        ``timeout`` is set from ``_llm_http_timeout_seconds()`` so Docker agents
        do not hit the OpenHands default 300s read timeout on slow generations.
    """
    llm_timeout: int = _llm_http_timeout_seconds()
    config: dict[str, object] = {
        "model": model_name,
        "base_url": base_url,
        "api_key": "not-needed",
        "timeout": llm_timeout,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 8192,
        "litellm_extra_body": {
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
        },
    }

    # temporary config file
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, config_file, indent=2)
    config_file.close()

    return config_file.name


def _server_ready_timeout_seconds() -> int:
    """Max seconds to wait for each model server (load + bind)."""
    raw: str = os.environ.get("SWEBENCH_SERVER_READY_TIMEOUT", "900").strip()
    try:
        parsed: int = int(raw)
    except ValueError:
        return 900
    return max(60, min(parsed, 7200))


def wait_for_server(
    port: int,
    timeout: Optional[int] = None,
    *,
    log_path: Optional[Path] = None,
) -> bool:
    """Wait until something accepts TCP on ``localhost:port``.

    ``start_server.py`` only opens the port after the model is loaded, so large
    checkpoints can take many minutes with no open port yet.

    Args:
        port: Port the server will bind.
        timeout: Seconds to wait (default from ``_server_ready_timeout_seconds()``).
        log_path: If set, print a short reminder every 30s with this path.

    Returns:
        True if the port became reachable in time.
    """
    limit: int = timeout if timeout is not None else _server_ready_timeout_seconds()
    start_time: float = time.time()
    next_progress: float = start_time + 30.0
    while time.time() - start_time < limit:
        if log_path is not None and time.time() >= next_progress:
            elapsed_s: int = int(time.time() - start_time)
            print(
                f"  ... still loading ({elapsed_s}s / {limit}s). "
                f"No port until load finishes — see: {log_path}"
            )
            next_progress = time.time() + 30.0
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result: int = sock.connect_ex(("localhost", port))
                if result == 0:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


def latest_context_token_count_from_server_log(
    server_log: Path,
    *,
    tail_bytes: int = 400_000,
) -> Optional[int]:
    """Parse the most recent ``Context tokens:`` line from a ``start_server`` log.

    ``ModelAdapterHF`` logs ``Context tokens: torch.Size([1, N])`` after truncation.

    Args:
        server_log: Path to ``server_gpu_*.log`` under the run output directory.
        tail_bytes: How many trailing bytes of the file to scan (large logs).

    Returns:
        Token count ``N`` from the last matching line, or None if not found / unreadable.
    """
    if not server_log.is_file():
        return None
    try:
        raw: bytes = server_log.read_bytes()
    except OSError:
        return None
    chunk: bytes = raw[-tail_bytes:] if len(raw) > tail_bytes else raw
    text: str = chunk.decode("utf-8", errors="replace")
    pattern: re.Pattern[str] = re.compile(
        r"Context tokens:\s*torch\.Size\(\[\s*(\d+)\s*,\s*(\d+)\s*\]\)"
    )
    last_n: Optional[int] = None
    for line in text.splitlines():
        m: Optional[re.Match[str]] = pattern.search(line)
        if m:
            last_n = int(m.group(2))
    return last_n


def start_servers(
    model_name: str,
    num_gpus: int,
    gpus_per_server: int,
    base_port: int,
    output_dir: str,
    gpu_offset: int = 0,
    max_memory_per_gpu_gib: Optional[float] = None,
    is_hybrid: bool = False,
    backend: str = "hf",
    vllm_max_model_len: int = 131072,
    vllm_gpu_memory_utilization: float = 0.90,
    vllm_dtype: str = "auto",
    vllm_language_model_only: bool = True,
    vllm_reasoning_parser: str = "qwen3",
    vllm_tool_call_parser: str = "qwen3_coder",
    vllm_enable_auto_tool_choice: bool = True,
) -> List[subprocess.Popen]:
    """Start one model-server process per replica, each using ``gpus_per_server`` GPUs.

    Args:
        model_name: Model identifier for local server startup.
        num_gpus: Total physical GPU count in this allocation (after ``gpu_offset``).
        gpus_per_server: GPUs visible to each server (``CUDA_VISIBLE_DEVICES`` length).
        base_port: Port for replica 0; replica ``k`` uses ``base_port + k``.
        output_dir: Directory for server logs.
        gpu_offset: First physical GPU index (replicas use contiguous blocks).
        max_memory_per_gpu_gib: Optional cap passed to HF ``start_server.py`` for
            ``device_map=auto`` (recommended on 80GB cards for large models).
        is_hybrid: If True, pass ``--is-hybrid`` to ``start_server.py`` (Qwen3.5-style
            hybrid linear attention in adapter paths).
        backend: ``hf`` (scripts/start_server.py) or ``vllm`` (vllm serve).
        vllm_max_model_len: vLLM ``--max-model-len``.
        vllm_gpu_memory_utilization: vLLM ``--gpu-memory-utilization``.
        vllm_dtype: vLLM ``--dtype`` value.
        vllm_language_model_only: Add ``--language-model-only`` for text-only SWE tasks.
        vllm_reasoning_parser: vLLM ``--reasoning-parser``.
        vllm_tool_call_parser: vLLM ``--tool-call-parser``.
        vllm_enable_auto_tool_choice: Add ``--enable-auto-tool-choice``.

    Returns:
        List of server ``Popen`` objects (length ``num_gpus // gpus_per_server``).
    """
    servers: List[subprocess.Popen] = []
    server_logs: List[Path] = []
    num_servers: int = num_gpus // gpus_per_server

    for server_rank in range(num_servers):
        first_physical: int = gpu_offset + server_rank * gpus_per_server
        visible_list: List[str] = [
            str(first_physical + j) for j in range(gpus_per_server)
        ]
        visible: str = ",".join(visible_list)
        port: int = base_port + server_rank
        log_file: Path = Path(output_dir) / f"server_rank_{server_rank}.log"

        env: Dict[str, str] = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible
        metrics_dir: Path = Path(output_dir) / "metrics" / f"replica_{server_rank}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        env["SAH_METRICS_LOG_DIR"] = str(metrics_dir)

        project_root: Path = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(project_root)

        print(
            f"Starting {backend} server replica {server_rank} on port {port} "
            f"(CUDA_VISIBLE_DEVICES={visible})..."
        )

        if backend == "hf":
            server_python: str = resolve_server_python()
            cmd: List[str] = [
                server_python,
                "scripts/start_server.py",
                "--device-map",
                "auto",
            ]
            if max_memory_per_gpu_gib is not None:
                cmd.extend(
                    ["--max-memory-per-gpu-gib", f"{float(max_memory_per_gpu_gib):g}"]
                )
            if is_hybrid:
                cmd.append("--is-hybrid")
            cmd.extend([model_name, str(port)])
        else:
            cmd = [
                resolve_vllm_bin(),
                "serve",
                model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(gpus_per_server),
                "--max-model-len",
                str(vllm_max_model_len),
                "--gpu-memory-utilization",
                f"{float(vllm_gpu_memory_utilization):g}",
                "--dtype",
                vllm_dtype,
                "--trust-remote-code",
                "--reasoning-parser",
                vllm_reasoning_parser,
                "--tool-call-parser",
                vllm_tool_call_parser,
            ]
            if vllm_language_model_only:
                cmd.append("--language-model-only")
            if vllm_enable_auto_tool_choice:
                cmd.append("--enable-auto-tool-choice")

        with open(log_file, "w") as log_handle:
            server_process: subprocess.Popen = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=project_root,
            )

        servers.append(server_process)
        server_logs.append(log_file)

        print(f"Waiting for replica {server_rank} to be ready on port {port}...")
        print(
            "  Note: Port stays closed until the model finishes loading "
            "(often several minutes for multi-billion-parameter models)."
        )
        print(f"  Log: {log_file}")
        if wait_for_server(port, log_path=log_file):
            print(f"✓ Replica {server_rank} ready on port {port}")
        else:
            print(
                f"✗ Replica {server_rank} failed to respond on port {port} within "
                f"{_server_ready_timeout_seconds()}s — check {log_file}"
            )

    print("\nFinal server status check:")
    all_ready: bool = True
    for server_rank, (server, log_file) in enumerate(zip(servers, server_logs)):
        if server.poll() is None:
            port = base_port + server_rank
            if wait_for_server(port, timeout=5):
                print(
                    f"✓ Replica {server_rank} running and responding "
                    f"(PID: {server.pid})"
                )
            else:
                print(
                    f"⚠ Replica {server_rank} running but not responding on port {port}"
                )
                all_ready = False
        else:
            print(f"✗ Replica {server_rank} failed to start. Check {log_file}")
            all_ready = False

    if not all_ready:
        print("\nWarning: Not all servers are ready. Inference may fail.")
        print("Check server logs for details.")

    return servers


def run_inference_jobs(
    instances_files: List[str],
    llm_configs: List[str],
    output_dir: str,
    model_name: str,
    num_workers: int,
) -> List[subprocess.Popen]:
    """Run inference jobs in parallel."""
    inference_jobs = []

    for gpu_id, (instances_file, config_file) in enumerate(zip(instances_files, llm_configs)):
        output_subdir = Path(output_dir) / f"gpu_{gpu_id}"
        output_subdir.mkdir(exist_ok=True)

        log_file = output_subdir / "inference.log"

        print(f"Starting inference job for GPU {gpu_id}...")

        # Run inference
        # Use absolute path for output_dir to ensure it goes exactly where we want
        env = os.environ.copy()
        env.setdefault('LITELLM_TIMEOUT', '5400')
        # Match benchmarks default in ``fake_user_response`` (90 min per run segment).
        env.setdefault('CONVERSATION_TIMEOUT', '5400')
        
        # pass@1-style: one critic pass (--n-critic-runs 1); no extra runs on failures.
        # (--max-attempts is not a valid swebench-infer flag; use --n-critic-runs instead.)
        cmd = [
            'uv', 'run', 'swebench-infer',
            config_file,
            '--dataset', 'princeton-nlp/SWE-bench_Verified',
            '--split', 'test',
            '--max-iterations', '300',
            '--n-critic-runs', '1',
            '--max-retries', '0',
            '--workspace', 'docker',
            '--select', instances_file,
            '--output-dir', str(output_subdir.absolute()),
            '--note', f'gpu_{gpu_id}_run',
            '--num-workers', str(num_workers),
        ]

        with open(log_file, 'w') as f:
            inference_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent / 'benchmarks'  # benchmarks directory
            )

        inference_jobs.append(inference_process)

    return inference_jobs


def collect_outputs(output_dir: str, num_parallel_tracks: int) -> tuple[int, int]:
    """Collect all outputs into a single output.jsonl file.

    Successful trajectories live in each run's ``output.jsonl`` (aggregated from
    critic attempts without ``error``). Failed runs often only appear under
    ``output_errors.jsonl`` or ``output.critic_attempt_*.jsonl``; those are
    merged into ``output_errors.jsonl`` at the top level for inspection.

    Args:
        output_dir: Run output root (contains ``gpu_0``, ``gpu_1``, ...).
        num_parallel_tracks: Number of parallel inference shards (one per server
            replica; equals ``num_gpus // gpus_per_server`` when using multi-GPU
            servers).

    Returns:
        Tuple of ``(total_instance_rows, num_shards_with_output_jsonl)``.
    """
    output_dir = Path(output_dir)
    combined_output = output_dir / "output.jsonl"
    combined_errors = output_dir / "output_errors.jsonl"
    combined_cost = output_dir / "cost_report.jsonl"

    print(f"Collecting outputs into {combined_output}...")

    total_instances = 0
    successful_outputs = 0
    total_failed_lines: int = 0

    with open(combined_output, 'w') as outfile, open(combined_errors, 'w') as errfile:
        for gpu_id in range(num_parallel_tracks):
            gpu_subdir = output_dir / f"gpu_{gpu_id}"

            # Find any output.jsonl file in the structured subdirectories
            found_output: Optional[Path] = None
            for p in gpu_subdir.glob("**/output.jsonl"):
                found_output = p
                break

            if found_output and found_output.exists():
                gpu_count: int = 0
                with open(found_output, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                        gpu_count += 1
                rel = found_output.relative_to(output_dir)
                print(f"✓ GPU {gpu_id}: {gpu_count} successful instance(s) (from {rel})")
                if gpu_count == 0:
                    attempt_path: Optional[Path] = None
                    for p in found_output.parent.glob("output.critic_attempt_*.jsonl"):
                        attempt_path = p
                        break
                    err_sidecar: Path = found_output.parent / "output_errors.jsonl"
                    if attempt_path and attempt_path.exists():
                        with open(attempt_path, 'r', encoding='utf-8') as af:
                            n_attempt: int = sum(
                                1 for line in af if line.strip()
                            )
                        print(
                            f"  ℹ GPU {gpu_id}: {n_attempt} row(s) in {attempt_path.name} "
                            f"(all had errors — see {err_sidecar.name} or "
                            f"gpu_{gpu_id}/inference.log)"
                        )
                successful_outputs += 1
                total_instances += gpu_count
            else:
                print(f"✗ GPU {gpu_id}: No output.jsonl found in {gpu_subdir}")

            # Merge per-GPU error JSONL for a single top-level artifact
            for err_path in gpu_subdir.glob("**/output_errors.jsonl"):
                with open(err_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            errfile.write(line)
                            total_failed_lines += 1

    print(f"Total successful instances collected: {total_instances}")
    if total_failed_lines > 0:
        print(
            f"Total failed-instance records merged to {combined_errors.name}: "
            f"{total_failed_lines} (check Docker/buildx and inference logs if non-zero)"
        )

    # Also collect cost reports if they exist
    cost_reports_found = 0
    with open(combined_cost, 'w') as outfile:
        for gpu_id in range(num_parallel_tracks):
            gpu_subdir = output_dir / f"gpu_{gpu_id}"
            for cost_report in gpu_subdir.glob("**/cost_report.jsonl"):
                if cost_report.exists():
                    with open(cost_report, 'r') as infile:
                        for line in infile:
                            outfile.write(line)
                    cost_reports_found += 1
                    break

    if cost_reports_found > 0:
        print(f"✓ Cost reports collected from {cost_reports_found} GPUs")
    else:
        print("ℹ No cost reports found")

    return total_instances, successful_outputs


def cleanup_temp_files(temp_files: List[str]):
    """Clean up temporary files."""
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass


def validate_environment():
    """Validate that required dependencies are available."""
    print("Validating environment...")

    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "sparse_attention_hub").exists():
        print("Error: Must run from sparse-attention-hub project root")
        sys.exit(1)

    # Check if benchmarks directory exists
    if not (project_root / "benchmarks").exists():
        print("Error: benchmarks directory not found. Run from project root.")
        sys.exit(1)

    # Check if uv is available
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        print("✓ uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: uv is not available. Please install uv.")
        sys.exit(1)

    # Check if docker is available
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
        print("✓ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Docker not found. Make sure Docker is running for SWE-Bench inference.")

    print("✓ Environment validation complete")


def main():
    parser = argparse.ArgumentParser(description="Distributed SWE-Bench Inference")
    parser.add_argument('--instances_file', required=True, help='Path to instances text file')
    parser.add_argument('--model_name', required=True, help='Model name (e.g., Qwen/Qwen3-Coder-30B-A3B-Instruct)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=8,
        help='Total physical GPUs on this node for this run (default: 8).',
    )
    parser.add_argument(
        '--gpus-per-server',
        type=int,
        default=1,
        dest='gpus_per_server',
        help=(
            'GPUs assigned to each HF server process (default: 1). Must divide '
            '--num_gpus. Example: --num_gpus 8 --gpus-per-server 4 → two replicas, '
            'each with CUDA_VISIBLE_DEVICES spanning four cards and device_map=auto.'
        ),
    )
    parser.add_argument(
        '--max-memory-per-gpu-gib',
        type=float,
        default=None,
        dest='max_memory_per_gpu_gib',
        help=(
            'Optional HuggingFace max_memory per visible GPU (GiB) for multi-GPU '
            'servers; forwarded to start_server.py. Recommended for large models on '
            '80GB cards (e.g. 72).'
        ),
    )
    parser.add_argument('--base_port', type=int, default=4000, help='Base port number (default: 4000)')
    parser.add_argument(
        '--gpu_offset',
        type=int,
        default=0,
        help='Offset to add to GPU indices (e.g., 4 to use GPUs 4,5,6,7 instead of 0,1,2,3)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=30,
        dest='num_workers',
        help=(
            'Parallel SWE-bench instances per GPU process (default: 30, capped by the harness). '
            'Use 1 on large models / tight VRAM so only one trajectory hits each local server.'
        ),
    )
    parser.add_argument(
        '--is-hybrid',
        action='store_true',
        help=(
            'Forward to start_server.py as --is-hybrid (hybrid linear-attention models '
            'such as Qwen3.5).'
        ),
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='hf',
        choices=('hf', 'vllm'),
        help=(
            "Model server backend. 'hf' launches scripts/start_server.py; "
            "'vllm' launches `vllm serve`. Default: hf."
        ),
    )
    parser.add_argument(
        '--vllm-max-model-len',
        type=int,
        default=131072,
        help='vLLM only: value for --max-model-len (default: 131072).',
    )
    parser.add_argument(
        '--vllm-gpu-memory-utilization',
        type=float,
        default=0.90,
        help='vLLM only: value for --gpu-memory-utilization (default: 0.90).',
    )
    parser.add_argument(
        '--vllm-dtype',
        type=str,
        default='auto',
        help='vLLM only: value for --dtype (default: auto).',
    )
    parser.add_argument(
        '--vllm-language-model-only',
        action='store_true',
        default=True,
        help='vLLM only: pass --language-model-only (enabled by default).',
    )
    parser.add_argument(
        '--no-vllm-language-model-only',
        action='store_false',
        dest='vllm_language_model_only',
        help='vLLM only: disable --language-model-only.',
    )
    parser.add_argument(
        '--vllm-enable-auto-tool-choice',
        action='store_true',
        default=True,
        help='vLLM only: pass --enable-auto-tool-choice (enabled by default).',
    )
    parser.add_argument(
        '--no-vllm-enable-auto-tool-choice',
        action='store_false',
        dest='vllm_enable_auto_tool_choice',
        help='vLLM only: disable --enable-auto-tool-choice.',
    )
    parser.add_argument(
        '--vllm-reasoning-parser',
        type=str,
        default='qwen3',
        help='vLLM only: value for --reasoning-parser (default: qwen3).',
    )
    parser.add_argument(
        '--vllm-tool-call-parser',
        type=str,
        default='qwen3_coder',
        help='vLLM only: value for --tool-call-parser (default: qwen3_coder).',
    )
    parser.add_argument('--skip_validation', action='store_true', help='Skip environment validation')

    args = parser.parse_args()
    llm_host: str = resolve_llm_host()

    if args.num_gpus < 1:
        print("Error: --num_gpus must be >= 1")
        sys.exit(1)
    if args.gpus_per_server < 1:
        print("Error: --gpus-per-server must be >= 1")
        sys.exit(1)
    if args.num_gpus % args.gpus_per_server != 0:
        print(
            f"Error: --num_gpus ({args.num_gpus}) must be divisible by "
            f"--gpus-per-server ({args.gpus_per_server})."
        )
        sys.exit(1)
    num_servers: int = args.num_gpus // args.gpus_per_server

    # Validate environment
    if not args.skip_validation:
        validate_environment()

    # Validate inputs
    instances_file = Path(args.instances_file)
    if not instances_file.exists():
        print(f"Error: Instances file {instances_file} does not exist")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output directory is empty (optional warning)
    existing_files = list(output_dir.glob("*"))
    if existing_files:
        print(f"Warning: Output directory {output_dir} is not empty.")
        print(f"Existing files: {[f.name for f in existing_files[:5]]}")
        if len(existing_files) > 5:
            print(f"... and {len(existing_files) - 5} more")
        print("Continuing anyway...")

    print(f"Starting distributed SWE-Bench inference...")
    print(f"Instances file: {instances_file}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"GPUs per server: {args.gpus_per_server}")
    print(f"Server replicas: {num_servers}")
    print(f"GPU offset: {args.gpu_offset} (using GPUs {args.gpu_offset}-{args.gpu_offset + args.num_gpus - 1})")
    print(f"Base port: {args.base_port}")
    print(f"swebench-infer --num-workers: {args.num_workers}")
    print(f"Backend: {args.backend}")
    print(f"--is-hybrid (adapter): {args.is_hybrid}")
    print(f"LLM host (for Docker -> host): {llm_host}")
    if args.backend == "vllm":
        print(
            "vLLM server flags: "
            f"max_model_len={args.vllm_max_model_len}, "
            f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}, "
            f"dtype={args.vllm_dtype}, "
            f"language_model_only={args.vllm_language_model_only}, "
            f"reasoning_parser={args.vllm_reasoning_parser}, "
            f"tool_call_parser={args.vllm_tool_call_parser}, "
            f"enable_auto_tool_choice={args.vllm_enable_auto_tool_choice}"
        )

    # Track processes for cleanup
    servers = []
    inference_jobs = []
    temp_files = []

    try:
        # Step 1: Split instances file
        print("\n=== Step 1: Splitting instances ===")
        split_files = split_instances_file(str(instances_file), num_servers)
        temp_files.extend(split_files)

        # Step 2: Start servers
        print("\n=== Step 2: Starting servers ===")
        if args.backend == "hf":
            server_py: str = resolve_server_python()
            print(f"HF server Python (each start_server child): {server_py}")
            if server_py == sys.executable:
                print(
                    "  Tip: if the server log shows CUDA/driver errors or "
                    "'--device-map auto requires CUDA', run from a GPU-matched env, e.g.\n"
                    "    conda activate swebench311 && "
                    "export SPARSE_ATTENTION_SERVER_PYTHON=$(which python)\n"
                    "  then re-run this script (same shell)."
                )
            else:
                print(f"  (launcher Python was: {sys.executable})")
        else:
            print(f"vLLM binary: {resolve_vllm_bin()}")

        # Strip provider prefix if present for server startup
        # (e.g., 'openai/Qwen/...' -> 'Qwen/...')
        server_model_name: str = strip_provider_prefix(args.model_name)
        if server_model_name != args.model_name:
            print(f"ℹ Stripped provider prefix for server startup: {server_model_name}")
        
        servers = start_servers(
            server_model_name,
            args.num_gpus,
            args.gpus_per_server,
            args.base_port,
            str(output_dir),
            args.gpu_offset,
            args.max_memory_per_gpu_gib,
            args.is_hybrid,
            args.backend,
            args.vllm_max_model_len,
            args.vllm_gpu_memory_utilization,
            args.vllm_dtype,
            args.vllm_language_model_only,
            args.vllm_reasoning_parser,
            args.vllm_tool_call_parser,
            args.vllm_enable_auto_tool_choice,
        )

        # Step 3: Create LLM configs
        print("\n=== Step 3: Creating LLM configs ===")
        llm_configs = []
        for replica_id in range(num_servers):
            port: int = args.base_port + replica_id
            base_url = f"http://{llm_host}:{port}/v1"
            config_file = create_llm_config(base_url, args.model_name)
            llm_configs.append(config_file)
            temp_files.append(config_file)

        # Step 4: Run inference jobs
        print("\n=== Step 4: Running inference jobs ===")
        inference_jobs = run_inference_jobs(
            split_files,
            llm_configs,
            str(output_dir),
            args.model_name,
            args.num_workers,
        )

        # Step 5: Wait for completion
        print("\n=== Step 5: Waiting for completion ===")
        print("Monitoring inference jobs...")

        while True:
            all_done = True
            for gpu_id, job in enumerate(inference_jobs):
                if job.poll() is None:
                    all_done = False
                    print(f"GPU {gpu_id}: Running (PID: {job.pid})")
                    server_log: Path = output_dir / f"server_rank_{gpu_id}.log"
                    if args.backend == "hf":
                        n_tokens: Optional[int] = latest_context_token_count_from_server_log(
                            server_log
                        )
                        if n_tokens is not None:
                            print(
                                f"        latest request context (from {server_log.name}): "
                                f"{n_tokens} tokens"
                            )
                        else:
                            print(
                                f"        latest request context (from {server_log.name}): "
                                f"(no Context tokens line yet)"
                            )
                    else:
                        print(f"        server log: {server_log.name}")
                else:
                    exit_code = job.returncode
                    status = "✓" if exit_code == 0 else f"✗ (exit code: {exit_code})"
                    print(f"GPU {gpu_id}: {status}")

            if all_done:
                break

            time.sleep(30)

        # Step 6: Collect outputs
        print("\n=== Step 6: Collecting outputs ===")
        total_instances, successful_outputs = collect_outputs(str(output_dir), num_servers)

        print("\n=== Inference complete! ===")
        print(f"✓ Combined output saved to: {output_dir}/output.jsonl")
        print(f"✓ Total successful trajectories: {total_instances}")
        print(
            f"✓ Shards with a run directory (output.jsonl present): "
            f"{successful_outputs}/{num_servers}"
        )
        print(f"✓ Individual shard outputs in: {output_dir}/gpu_*/")
        merged_errors: Path = output_dir / "output_errors.jsonl"
        if merged_errors.exists() and merged_errors.stat().st_size > 0:
            print(f"✓ Failed runs (if any) merged to: {merged_errors}")

        if successful_outputs < num_servers:
            print(
                f"⚠ Only {successful_outputs}/{num_servers} shards produced outputs. "
                "Check logs for failures."
            )
        if total_instances == 0 and successful_outputs > 0:
            print(
                "⚠ No successful trajectories — instances likely failed before the agent "
                "finished (often Docker/buildx). See output_errors.jsonl and gpu_*/inference.log."
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        print("\n=== Cleanup ===")

        # Terminate servers
        print("Terminating servers...")
        for replica_id, server in enumerate(servers):
            try:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=10)
                    print(f"✓ Replica {replica_id} server terminated")
                else:
                    print(f"Replica {replica_id} server already stopped")
            except Exception:
                try:
                    server.kill()
                    print(f"✓ Replica {replica_id} server killed")
                except Exception:
                    print(f"✗ Could not terminate replica {replica_id} server")

        # Clean up temp files
        cleanup_temp_files(temp_files)
        print("✓ Temporary files cleaned up")


if __name__ == "__main__":
    main()

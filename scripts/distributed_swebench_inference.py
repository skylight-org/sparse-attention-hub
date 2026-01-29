#!/usr/bin/env python3
"""
Distributed SWE-Bench Inference Script

This script distributes SWE-Bench inference across multiple GPUs on a single node.
It splits instances, starts multiple servers with sparse attention, and runs parallel inference.

Usage:
    python scripts/distributed_swebench_inference.py \
        --instances_file path/to/instances.txt \
        --model_name openai/Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --output_dir /path/to/output \
        --num_gpus 8 \
        --base_port 4000
"""

#pd (things to look at later if we want to optimize)
#i think we need to print the config in the terminal when we launch this, rn need to do some tail command business
#look into slurm? but depending on optimization this could be ok
# look into max retries flag
#memmory requirements for this? on 80gb h100 am running close to memory

#tested command:
# python scripts/distributed_swebench_inference.py   --instances_file scripts/example_instances.txt   --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct   --output_dir /tmp/test_distributed_output   --num_gpus 2   --base_port 4000


import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
import socket


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


def create_llm_config(base_url: str, model_name: str) -> str:
    """Create LLM config file for a specific server."""
    config = {
        "model": model_name,
        "base_url": base_url,
        "api_key": "not-needed",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 65536,
        "litellm_extra_body": {
            "repetition_penalty": 1.05
        }
    }

    # temporary config file
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, config_file, indent=2)
    config_file.close()

    return config_file.name


def wait_for_server(port: int, timeout: int = 120) -> bool:
    """Wait for server to be ready on given port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except:
            pass
        time.sleep(2)
    return False


def start_servers(model_name: str, num_gpus: int, base_port: int, output_dir: str) -> List[subprocess.Popen]:
    """Start multiple servers, one per GPU."""
    servers = []
    server_logs = []

    for gpu_id in range(num_gpus):
        port = base_port + gpu_id
        log_file = Path(output_dir) / f"server_gpu_{gpu_id}.log"

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Set PYTHONPATH to ensure imports work in subprocess
        project_root = Path(__file__).parent.parent
        env['PYTHONPATH'] = str(project_root)

        print(f"Starting server for GPU {gpu_id} on port {port}...")

        # Start server process
        cmd = [
            sys.executable,
            'scripts/start_server.py',
            model_name,
            'sparse',  # Use sparse attention
            str(port)
        ]

        with open(log_file, 'w') as f:
            server_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent  #project root
            )

        servers.append(server_process)
        server_logs.append(log_file)

        # Wait for server to be ready
        print(f"Waiting for GPU {gpu_id} server to be ready on port {port}...")
        if wait_for_server(port, timeout=120):
            print(f"✓ GPU {gpu_id} server ready on port {port}")
        else:
            print(f"✗ GPU {gpu_id} server failed to respond on port {port}")

    # Final verification
    print("\nFinal server status check:")
    all_ready = True
    for gpu_id, (server, log_file) in enumerate(zip(servers, server_logs)):
        if server.poll() is None:
            port = base_port + gpu_id
            if wait_for_server(port, timeout=5):
                print(f"✓ GPU {gpu_id} server running and responding (PID: {server.pid})")
            else:
                print(f"⚠ GPU {gpu_id} server running but not responding on port {port}")
                all_ready = False
        else:
            print(f"✗ GPU {gpu_id} server failed to start. Check {log_file}")
            all_ready = False

    if not all_ready:
        print("\nWarning: Not all servers are ready. Inference may fail.")
        print("Check server logs for details.")

    return servers


def run_inference_jobs(
    instances_files: List[str],
    llm_configs: List[str],
    output_dir: str,
    model_name: str
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
        cmd = [
            'uv', 'run', 'swebench-infer',
            config_file,
            '--dataset', 'princeton-nlp/SWE-bench_Verified',
            '--split', 'test',
            '--max-iterations', '100',
            '--workspace', 'docker',
            '--select', instances_file,
            '--output-dir', str(output_subdir.absolute()),
            '--note', f'gpu_{gpu_id}_run'
        ]

        with open(log_file, 'w') as f:
            inference_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent / 'benchmarks'  # benchmarks directory
            )

        inference_jobs.append(inference_process)

    return inference_jobs


def collect_outputs(output_dir: str, num_gpus: int):
    """Collect all outputs into a single output.jsonl file."""
    output_dir = Path(output_dir)
    combined_output = output_dir / "output.jsonl"
    combined_cost = output_dir / "cost_report.jsonl"

    print(f"Collecting outputs into {combined_output}...")

    total_instances = 0
    successful_outputs = 0

    with open(combined_output, 'w') as outfile:
        for gpu_id in range(num_gpus):
            gpu_subdir = output_dir / f"gpu_{gpu_id}"
            
            # Find any output.jsonl file in the structured subdirectories
            found_output = None
            for p in gpu_subdir.glob("**/output.jsonl"):
                found_output = p
                break
            
            if found_output and found_output.exists():
                gpu_count = 0
                with open(found_output, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                        gpu_count += 1
                print(f"✓ GPU {gpu_id}: {gpu_count} instances collected (from {found_output.relative_to(output_dir)})")
                successful_outputs += 1
                total_instances += gpu_count
            else:
                print(f"✗ GPU {gpu_id}: No output file found in {gpu_subdir}")

    print(f"Total instances collected: {total_instances}")

    # Also collect cost reports if they exist
    cost_reports_found = 0
    with open(combined_cost, 'w') as outfile:
        for gpu_id in range(num_gpus):
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
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs (default: 8)')
    parser.add_argument('--base_port', type=int, default=4000, help='Base port number (default: 4000)')
    parser.add_argument('--skip_validation', action='store_true', help='Skip environment validation')

    args = parser.parse_args()

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
    print(f"Base port: {args.base_port}")

    # Track processes for cleanup
    servers = []
    inference_jobs = []
    temp_files = []

    try:
        # Step 1: Split instances file
        print("\n=== Step 1: Splitting instances ===")
        split_files = split_instances_file(str(instances_file), args.num_gpus)
        temp_files.extend(split_files)

        # Step 2: Start servers
        print("\n=== Step 2: Starting servers ===")
        
        # Strip provider prefix if present for server startup
        # (e.g., 'openai/Qwen/...' -> 'Qwen/...')
        server_model_name = args.model_name
        if '/' in server_model_name:
            parts = server_model_name.split('/')
            # Common LiteLLM/OpenHands provider prefixes
            if parts[0] in ['openai', 'anthropic', 'google', 'huggingface', 'litellm']:
                server_model_name = '/'.join(parts[1:])
                print(f"ℹ Stripped provider prefix for server startup: {server_model_name}")
        
        servers = start_servers(server_model_name, args.num_gpus, args.base_port, str(output_dir))

        # Step 3: Create LLM configs
        print("\n=== Step 3: Creating LLM configs ===")
        llm_configs = []
        for gpu_id in range(args.num_gpus):
            port = args.base_port + gpu_id
            # OpenHands in Docker uses 172.17.0.1 to reach host on this machine(?) maybe fix globally later
            base_url = f"http://172.17.0.1:{port}/v1"
            config_file = create_llm_config(base_url, args.model_name)
            llm_configs.append(config_file)
            temp_files.append(config_file)

        # Step 4: Run inference jobs
        print("\n=== Step 4: Running inference jobs ===")
        inference_jobs = run_inference_jobs(split_files, llm_configs, str(output_dir), args.model_name)

        # Step 5: Wait for completion
        print("\n=== Step 5: Waiting for completion ===")
        print("Monitoring inference jobs...")

        while True:
            all_done = True
            for gpu_id, job in enumerate(inference_jobs):
                if job.poll() is None:
                    all_done = False
                    print(f"GPU {gpu_id}: Running (PID: {job.pid})")
                else:
                    exit_code = job.returncode
                    status = "✓" if exit_code == 0 else f"✗ (exit code: {exit_code})"
                    print(f"GPU {gpu_id}: {status}")

            if all_done:
                break

            time.sleep(30)

        # Step 6: Collect outputs
        print("\n=== Step 6: Collecting outputs ===")
        total_instances, successful_outputs = collect_outputs(str(output_dir), args.num_gpus)

        print("\n=== Inference complete! ===")
        print(f"✓ Combined output saved to: {output_dir}/output.jsonl")
        print(f"✓ Total instances processed: {total_instances}")
        print(f"✓ GPUs with outputs: {successful_outputs}/{args.num_gpus}")
        print(f"✓ Individual GPU outputs in: {output_dir}/gpu_*/")

        if successful_outputs < args.num_gpus:
            print(f"⚠ Only {successful_outputs}/{args.num_gpus} GPUs produced outputs. Check logs for failures.")

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
        for gpu_id, server in enumerate(servers):
            try:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=10)
                    print(f"✓ GPU {gpu_id} server terminated")
                else:
                    print(f"GPU {gpu_id} server already stopped")
            except:
                try:
                    server.kill()
                    print(f"✓ GPU {gpu_id} server killed")
                except:
                    print(f"✗ Could not terminate GPU {gpu_id} server")

        # Clean up temp files
        cleanup_temp_files(temp_files)
        print("✓ Temporary files cleaned up")


if __name__ == "__main__":
    main()

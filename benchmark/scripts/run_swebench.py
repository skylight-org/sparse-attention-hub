import os
import sys
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

def main():
    # Setup configs
    adapter_cfg = AdapterConfig(
        model_kwargs={"torch_dtype": torch.bfloat16},
        swebench_api_endpoint="http://localhost:8000/submit" #irrelevantendpoint of the running SWE-bench API server, irrelevant for now
    )

    executor = BenchmarkExecutor(
        gpu_ids=[0, 1], 
        max_concurrent_runs=2,
        base_result_dir="./results"
    )
    
    # Split the 300 samples into 2 chunks of 150 for parallel execution on 2 GPUs
    subsets = ["swe-bench_verified:0:150", "swe-bench_verified:150:300"]

    #trigger worker run inference call api
    executor.run_benchmark_matrix(
        model_names=["Qwen/Qwen2.5-Coder-32B-Instruct"],
        sparse_attention_configs=[("dense", None)],
        benchmark_configs=[BenchmarkConfig(benchmark_name="swebench", subsets=subsets)],
        adapter_config=adapter_cfg,
        generation_kwargs={"max_new_tokens": 8192},
        request_kwargs={"max_turns": 30}
    )

if __name__ == "__main__":
    main()

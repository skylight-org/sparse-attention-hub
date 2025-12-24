import torch
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Setup configs
adapter_cfg = AdapterConfig(
    model_kwargs={"torch_dtype": torch.bfloat16},
    swebench_api_endpoint="http://localhost:8000/submit" #endpoint of the running SWE-bench API server
)

executor = BenchmarkExecutor(
    gpu_ids=[0, 1], 
    max_concurrent_runs=2,
    base_result_dir="./results"
)

#trigger worker run inference call api
executor.run_benchmark_matrix(
    model_names=["meta-llama/Llama-3.2-1B-Instruct"],
    sparse_attention_configs=[("dense", None)],
    benchmark_configs=[BenchmarkConfig(benchmark_name="swebench")],
    adapter_config=adapter_cfg
)
#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.basic_fixed import (
    LocalMaskerConfig, SinkMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.pq_top_k import (
    PQCacheConfig
)


#our qwen 2.5 models to benchmark
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

#benchmarking config 
BENCHMARKS = [
    BenchmarkConfig(benchmark_name="ruler32k", subsets=["niah_single_1", "fwe"]),
]
#just these two datasets for testing

#attn config
SPARSE_CONFIGS = [
    ("dense", None), #baseline
    ("pqcache", ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        PQCacheConfig(
            heavy_size=0.1,
            pq_group_factor=2,
            pq_bits=6,
            kmeans_iter=10,
            init_offset=128,
            metric="euclidean",
        ),
    ])),
]

#adapter settings for Qwen 2.5
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa", #check here
        "trust_remote_code": True,
    },
    tokenizer_kwargs={
        "padding_side": "left",
    }
)

#generation parameters, similar to full_benchmark.py
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 1.0,
}

#request parameters
REQUEST_KWARGS = {
    "max_context_length": 32768, # Adjust based on GPU memory
}

#system settings
GPUS = [0, 1] # List of GPU IDs to use
RESULT_DIR = "./benchmark_results/qwen25_suite"
MAX_CONCURRENT_RUNS = len(GPUS) #run one model per GPU


if __name__ == "__main__":
    print(f"🚀 Starting Qwen 2.5 Benchmark Suite")
    print(f"Models: {len(MODELS)}")
    print(f"Benchmarks: {len(BENCHMARKS)}")
    
    executor = BenchmarkExecutor(
        gpu_ids=GPUS,
        max_concurrent_runs=MAX_CONCURRENT_RUNS,
        base_result_dir=RESULT_DIR,
        enable_resumability=True, # Skip already completed runs
        verbose=True,
    )

    results = executor.run_benchmark_matrix(
        model_names=MODELS,
        sparse_attention_configs=SPARSE_CONFIGS,
        benchmark_configs=BENCHMARKS,
        adapter_config=ADAPTER_CONFIG,
        generation_kwargs=GENERATION_KWARGS,
        request_kwargs=REQUEST_KWARGS,
    )

    print(f"\n✅ Execution Completed. Results saved to {RESULT_DIR}")
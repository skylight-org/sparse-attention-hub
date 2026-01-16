#!/usr/bin/env python3
"""
Simple Benchmark Example

A beginner-friendly example showing how to run a basic benchmark comparison
between dense and sparse attention using the sparse-attention-hub framework.

This example uses the MockBenchmark (5 simple samples) for quick demonstration:
- Easy-to-understand reading comprehension questions
- Short contexts (<250 words each)
- Fast execution for testing and learning

Usage:
    python 04_simple_benchmark_example.py
"""

import os
import time
from pathlib import Path

import torch

# Ensure we're in the correct directory and add to Python path
import sys

# Change to directory two levels below current location
os.chdir('/scratch/sj157/sparse-attention-hub')
sys.path.insert(0, '/scratch/sj157/sparse-attention-hub')

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    DoubleSparsityTopKMaskerConfig
)

#from benchmark.longbench import LongBench
from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF

def main():
    # model_name = "mistralai/Ministral-3-8B-Instruct-2512"
    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
    ])
    
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=None,
        model_kwargs= {"torch_dtype": torch.bfloat16, "device_map": "auto"},
    )
    
    #benchmark = LongBench(['passage_retrieval_en'])
    benchmark = Ruler32K(['vt'])

    result_dir = Path("./test_results.vt.4096.2.2.q_proj/")
    result_dir.mkdir(exist_ok=True)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
            log_path=result_dir,
            enabled_metrics=[
                "research_attention_density",
                "research_attention_output_error",
            ],
        )
    metric_logger.flush()
    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 50, "max_context_length": 32000}, generation_kwargs={"max_new_tokens": 32})
    
if __name__ == "__main__":
    main() 

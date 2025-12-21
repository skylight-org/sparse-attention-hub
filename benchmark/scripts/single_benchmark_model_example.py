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
os.chdir('/data/apdesai/code/sparse-attention-hub')
sys.path.insert(0, '/data/apdesai/code/sparse-attention-hub')

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    QuestTopKMaskerConfig
)

#from benchmark.longbench import LongBench
from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF

def main():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = 0

    # sorted_channel_file is available in the author's repository
    # https://github.com/andy-yang-1/DoubleSparse/tree/main/config
    # TODO: is there a better way to use the paths in scripts?
    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128)
    ])
    
    print("  ✓ Loading model...")
     # use whichever is available
     # flash_attention_3 is for Hopper GPU
     # commonly flash_attention_2 is supported on other GPUs
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        device=device
    )
    
    #benchmark = LongBench(['passage_retrieval_en'])
    benchmark = Ruler32K(['niah_multikey_2'])

    result_dir = Path("./test_results.4B/")
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
    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 100, "max_context_length": 1000000}, generation_kwargs={"max_new_tokens": 500})
    
if __name__ == "__main__":
    main() 

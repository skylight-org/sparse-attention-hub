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

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Change to repository root based on script location
repo_root: Path = Path(__file__).resolve().parents[2]
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

#from benchmark.longbench import LongBench
from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.adapters.utils.config import ModelServerConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import ( 
    OracleTopKConfig,
    OracleTopPMaskerConfig
)

def compute_micro_metric_averages(micro_metrics_path: Path) -> tuple[float | None, float | None]:
    """Compute average density and attention error from micro metrics."""
    if not micro_metrics_path.exists():
        return None, None

    density_values: list[float] = []
    error_values: list[float] = []

    with micro_metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            metric_name: str = str(record.get("metric", ""))
            value = record.get("value", None)
            if value is None:
                continue
            value_float: float = float(value)
            if metric_name == "research_attention_density":
                density_values.append(value_float)
            elif metric_name == "research_attention_output_error":
                error_values.append(value_float)

    avg_density: float | None = (
        sum(density_values) / len(density_values) if density_values else None
    )
    avg_error: float | None = (
        sum(error_values) / len(error_values) if error_values else None
    )
    return avg_density, avg_error


def read_overall_score(metrics_path: Path) -> float | None:
    """Read the overall score from a metrics.json file."""
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics: dict[str, Any] = json.load(file)

    overall_score = metrics.get("overall_score", None)
    if overall_score is None:
        return None

    return float(overall_score)


def main() -> None:
    """Run a basic sparse attention benchmark example."""
    model_name: str = "mistralai/Ministral-3-3B-Instruct-2512"
    device: int = 0
    heavy_sizes = [0.005]
    datasets = ["qa_1", "qa_2", "niah_multikey_2", "niah_multikey_3", "fwe", "vt"]

    for dataset in datasets:
        for heavy_size in heavy_sizes:
            print(f"\n=== Running for dataset={dataset}, heavy_size={heavy_size} ===")
            sparse_attention_config: ResearchAttentionConfig = ResearchAttentionConfig(masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                OracleTopKConfig(heavy_size=heavy_size),
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=0.01,
                    epsilon=0.4,
                    delta=0.4,
                    init_offset=128,
                    local_offset=128,
                )
            ])

            adapter: ModelAdapterHF = ModelAdapterHF(
                model_name=model_name,
                sparse_attention_config=sparse_attention_config,
                model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                device=device,
                model_registry_path="sparse_attention_hub/adapters/model_servers/model_registry.yaml",
            )

            benchmark: Ruler32K = Ruler32K([dataset])

            result_dir: Path = Path(f"./test_results.4B_{dataset}_{heavy_size}/")
            # Remove result directory if it exists
            if result_dir.exists() and result_dir.is_dir():
                for file_path in result_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        for sub_file in file_path.rglob("*"):
                            if sub_file.is_file():
                                sub_file.unlink()
                            elif sub_file.is_dir():
                                sub_file.rmdir()
                        file_path.rmdir()
                result_dir.rmdir()
            result_dir.mkdir(exist_ok=True)
            metric_logger: MicroMetricLogger = MicroMetricLogger()
            metric_logger.configure_logging(
                    log_path=result_dir,
                    enabled_metrics=[
                        "research_attention_density",
                        "research_attention_output_error",
                    ],
                )
            metric_logger.flush()
            benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 100, "max_context_length": 32000}, generation_kwargs={"max_new_tokens": 20})

            micro_metrics_path: Path = result_dir / "micro_metrics.jsonl"
            avg_density, avg_error = compute_micro_metric_averages(micro_metrics_path)
            metrics_path: Path = result_dir / "metrics.json"
            overall_score: float | None = read_overall_score(metrics_path)

            # Write results to a unique txt file
            result_txt = Path(f"results_{dataset}_{heavy_size}.txt")
            with result_txt.open("w", encoding="utf-8") as f:
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Heavy size: {heavy_size}\n")
                if avg_density is not None:
                    f.write(f"Average density: {avg_density:.6f}\n")
                else:
                    f.write("Average density: n/a (micro_metrics.jsonl missing or empty)\n")
                if avg_error is not None:
                    f.write(f"Average error: {avg_error:.6f}\n")
                else:
                    f.write("Average error: n/a (micro_metrics.jsonl missing or empty)\n")
                if overall_score is not None:
                    f.write(f"Overall score: {overall_score:.6f}\n")
                else:
                    f.write("Overall score: n/a (overall_score missing)\n")

            print(f"Results written to {result_txt}")
if __name__ == "__main__":
    main() 

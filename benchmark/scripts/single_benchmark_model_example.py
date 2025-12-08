#!/usr/bin/env python3
"""
Grid search over RACEBucketMasker hyperparameters on Ruler32K.

Order of execution:

  For each dataset in:
      vt, fwe, niah_multikey_1, niah_multikey_2, niah_multikey_3, qa_1, qa_2

    Run ALL hyperparameter configs:
      (K, L, top_t, heavy_size)

For each (dataset, config), this script:
  - builds a sparse attention config using Sink + Local + RACEBucketMasker
  - loads the model with that config
  - runs Ruler32K on that single dataset
  - reads metrics.json from:
        /home/ac508/sparse-attention-hub/test_results.5cpt.topk.2/metrics.json
  - prints and appends a line like:
        K=5 L=8  top_t=2 heavy=0.2 | qa_2=36.0

All files live in:
    /home/ac508/sparse-attention-hub/test_results.5cpt.topk.2/

  - metrics.json   (overwritten for each run)
  - results.txt    (accumulates all lines, grouped by dataset)
"""

import os
from pathlib import Path
import sys
import json
import itertools
import gc
import torch

# ---------------------------------------------------------------------
# Project setup: make sure we're inside sparse-attention-hub
# ---------------------------------------------------------------------
os.chdir("/scratch/sj157/sparse-attention-hub")
sys.path.insert(0, "/scratch/sj157/sparse-attention-hub")

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    BucketMaskerConfig,
)
from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF


# ---------------------------------------------------------------------
# Results: SAME DIRECTORY AS metrics.json
# ---------------------------------------------------------------------
RESULTS_ROOT = Path("/scratch/sj157/sparse-attention-hub/test_results.5cpt.topk.2")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_ROOT / "metrics.json"
LOG_PATH = RESULTS_ROOT / "results.txt"

# Datasets you want, processed ONE BY ONE (outer loop)
RULER_DATASETS = [
    # "vt",
    # "fwe",
    # "niah_multikey_2",
    "niah_multikey_3",
    # "qa_1",
    # "qa_2",
]


# ---------------------------------------------------------------------
# Logging helper: PRINT + FILE, with alignment
# ---------------------------------------------------------------------
def append_result_line(K, L, top_t, heavy_size, dataset_name, score_value):
    """
    Print and append one aligned line, e.g.

    K=5 L=8  top_t=2 heavy=0.2 | qa_2=36.0
    K=5 L=10 top_t=2 heavy=0.2 | qa_2=48.0
    """
    # L field padded so L=8 and L=10 align:
    L_field = f"L={L:<2}"  # width 2 after "L="

    line = (
        f"K={K} {L_field} top_t={top_t} heavy={heavy_size} | "
        f"{dataset_name}={score_value}"
    )

    # Print to terminal
    print(line)

    # Append to file
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------
# Run ONE (dataset, config) pair
# ---------------------------------------------------------------------
def run_single(dataset_name, K, L, top_t, heavy_size, device, model_name):
    adapter = None
    benchmark = None

    try:
        print(
            f"\n=== Dataset: {dataset_name} | "
            f"CONFIG: K={K}, L={L}, top_t={top_t}, heavy={heavy_size} ==="
        )

        # Build sparse attention config for this run
        sparse_attention_config = ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            BucketMaskerConfig(K=K, L=L, top_t=top_t, heavy_size=heavy_size),
        ])

        # Load model for this config
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            generate_kwargs={"max_new_tokens": 32},
            device=device,
        )

        # Single-dataset Ruler32K
        benchmark = Ruler32K([dataset_name])

        benchmark.run_benchmark(
            adapter,
            RESULTS_ROOT,
            request_kwargs={"max_requests": 50, "max_context_length": 32000},
        )

        # Read metrics.json for THIS dataset + config
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)

            task_scores = metrics.get("task_scores", {})
            if dataset_name in task_scores and len(task_scores[dataset_name]) > 0:
                metric_name = list(task_scores[dataset_name].keys())[0]
                score_value = task_scores[dataset_name][metric_name]
            else:
                score_value = "NaN"
        else:
            score_value = "NaN"

        append_result_line(K, L, top_t, heavy_size, dataset_name, score_value)

    finally:
        # Free memory for next run
        del benchmark
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main: OUTER LOOP = DATASET, INNER LOOP = CONFIGS
# ---------------------------------------------------------------------
def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = "cuda"

    # Start fresh log
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Hyperparameter grid
    K_list = [4]
    L_list = list(range(10, 89))
    top_t_list = [4, 5]
    heavy_list = [0.02]

    # OUTER: dataset
    # INNER: all hyperparameter configs
    for ds in RULER_DATASETS:
        print(f"\n\n============================")
        print(f"### DATASET: {ds}")
        print(f"============================")

        for K, L, top_t, heavy_size in itertools.product(
            K_list, L_list, top_t_list, heavy_list
        ):
            run_single(ds, K, L, top_t, heavy_size, device, model_name)


if __name__ == "__main__":
    main()
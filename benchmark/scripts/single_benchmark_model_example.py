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
    PQCacheConfig
)
from benchmark.ruler32k import Ruler32K
from benchmark.longbench import LongBench
from benchmark.longbenchv2 import LongBenchv2
from sparse_attention_hub.adapters import ModelAdapterHF

# IMPORTANT: for stats hook (requires you replaced bucket_top_k.py with the stats version)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.bucket_top_k import (
    BucketMasker,
)

# ---------------------------------------------------------------------
# Results: SAME DIRECTORY AS metrics.json
# ---------------------------------------------------------------------
RESULTS_ROOT = Path("/scratch/sj157/sparse-attention-hub/test_results.5cpt.topk.2")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_ROOT / "metrics.json"
LOG_PATH = RESULTS_ROOT / "results_4.txt"

# Datasets you want, processed ONE BY ONE (outer loop)
RULER_DATASETS = [
    # "cwe",
    # "fwe",
    # "niah_multikey_1",
    # "niah_multikey_2",
    # "niah_multikey_3",
    # "niah_multiquery",
    # "niah_multivalue",
    # "niah_single_1",
    # "niah_single_2",
    # "niah_single_3",
    # "qa_1",
    # "qa_2",
    # "vt",
]

LONGBENCH_DATASETS = [
    # "narrativeqa", 
    # "qasper", 
    # "multifieldqa_en", 
    # "multifieldqa_zh", 
    # "hotpotqa", 
    # "2wikimqa", 
    # "musique", 
    # "dureader", 
    # "gov_report", 
    # "qmsum", 
    # "multi_news", 
    # "vcsum", 
    # "trec", 
    # "triviaqa", 
    # "samsum", 
    # "lsht", 
    # "passage_count",
    # "passage_retrieval_en", 
    # "passage_retrieval_zh", 
    # "lcc", 
    "repobench-p",
    # "0shot",
    # "cot"
]

# ---------------------------------------------------------------------
# Helper: find BucketMasker instance inside adapter/model object graph
# ---------------------------------------------------------------------
def find_first_instance(root, cls, max_nodes=20000):
    """
    Best-effort DFS over Python object graph to find an instance of `cls`.
    Skips torch.Tensors to avoid exploding the search.
    """
    seen = set()
    stack = [root]
    n = 0

    while stack and n < max_nodes:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        n += 1

        if isinstance(obj, cls):
            return obj

        # Dive into attributes
        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                if isinstance(v, torch.Tensor):
                    continue
                stack.append(v)

        # Dive into common containers
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    continue
                stack.append(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                if isinstance(v, torch.Tensor):
                    continue
                stack.append(v)

    return None


# ---------------------------------------------------------------------
# Logging helper: PRINT + FILE, with alignment
# ---------------------------------------------------------------------
def append_result_line(K, L, top_t, heavy_size, dataset_name, score_value, cand_stats=None):
    """
    Example:
    K=4 L=10 top_t=4 heavy=0.2 | qa_1=64.0 | cand_avg=... cand_min=... cand_max=... | final_avg=...
    """
    L_field = f"L={L:<2}"  # aligns L=8 and L=10

    extra = ""
    if cand_stats is not None:
        extra = (
            f" | cand_avg={cand_stats['cand_avg']:.2f}"
            f" cand_min={int(cand_stats['cand_min'])}"
            f" cand_max={int(cand_stats['cand_max'])}"
            f" | final_avg={cand_stats['final_avg']:.2f}"
            f" final_min={int(cand_stats['final_min'])}"
            f" final_max={int(cand_stats['final_max'])}"
        )

    line = (
        f"K={K} {L_field} top_t={top_t} heavy={heavy_size} | "
        f"{dataset_name}={score_value}{extra}"
    )

    print(line)
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
        sparse_attention_config = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                BucketMaskerConfig(K=K, L=L, top_t=top_t, heavy_size=heavy_size),
                # PQCacheConfig(
                #     heavy_size=0.1,
                #     pq_group_factor=2,  # pq_sub_dim = head_dim // pq_group_factor (e.g., 128 // 2 = 64)
                #     pq_bits=6,
                #     kmeans_iter=10,
                #     init_offset=128,
                #     metric="euclidean",
                # )
            ]
        )

        # Load model for this config
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            generate_kwargs={"max_new_tokens": 32},
            device=device,
        )

        # Find your BucketMasker instance and reset stats for THIS run
        bm = find_first_instance(adapter, BucketMasker)
        if bm is None:
            print("[WARN] Could not find BucketMasker instance to collect candidate stats.")
        else:
            bm.reset_candidate_stats()

        # Single-dataset Ruler32K
        # benchmark = Ruler32K([dataset_name])
        benchmark = LongBench([dataset_name])

        benchmark.run_benchmark(
            adapter,
            RESULTS_ROOT,
            request_kwargs={"max_requests": 100, "max_context_length": 32000},
        )

        # Read metrics.json for THIS run (your existing logic)
        score_value = "NaN"

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)

            task_scores = metrics.get("task_scores", {})

            if isinstance(task_scores, dict) and dataset_name in task_scores:
                ds_score = task_scores[dataset_name]

                # LongBench style: scalar score
                if isinstance(ds_score, (int, float)):
                    score_value = ds_score

                # Ruler-style (or others): dict of metric_name -> score
                elif isinstance(ds_score, dict) and len(ds_score) > 0:
                    metric_name = next(iter(ds_score.keys()))
                    score_value = ds_score[metric_name]

                # Sometimes list/other containers
                else:
                    score_value = "NaN"

        # Pull candidate stats from BucketMasker
        cand_stats = None
        if bm is not None:
            cand_stats = bm.get_candidate_stats()

        append_result_line(
            K, L, top_t, heavy_size, dataset_name, score_value, cand_stats=cand_stats
        )

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
    device = "cuda"  # or "cuda:0"

    # Start fresh log
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Hyperparameter grid
    K_list = [8]
    L_list = [100, 95, 90, 85, 80, 75]
    top_t_list = [12, 20, 25, 28, 32]
    heavy_list = [512]

    for ds in LONGBENCH_DATASETS:
        print("\n\n============================")
        print(f"### DATASET: {ds}")
        print("============================")

        for K, L, top_t, heavy_size in itertools.product(
            K_list, L_list, top_t_list, heavy_list
        ):
            run_single(ds, K, L, top_t, heavy_size, device, model_name)


if __name__ == "__main__":
    main()
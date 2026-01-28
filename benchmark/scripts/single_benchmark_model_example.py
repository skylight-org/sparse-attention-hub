import os
from pathlib import Path
import sys
import json
import itertools
import gc
import torch

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
os.chdir("") # Put your repo root path here
sys.path.insert(0, "") 

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    SocketMaskerConfig,
    PQCacheConfig,
    QuestTopKMaskerConfig
)
from benchmark.ruler32k import Ruler32K
from benchmark.longbench import LongBench
from benchmark.loogle.loogle import Loogle
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import OracleTopKConfig

# ---------------------------------------------------------------------
# =======================
# USER EDIT SECTION
# =======================
# ---------------------------------------------------------------------
BENCHMARK_KIND = "longbench"  # "ruler" or "longbench" or "loogle"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-8B"

sys.path.insert(0, "") 
RESULTS_ROOT = Path("") # Put your results path here
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_ROOT / "metrics.json"
LOG_PATH = RESULTS_ROOT / "results_longbench_llama3-1b-socket-qwen-qasper.txt"

MAX_REQUESTS = 100
MAX_CONTEXT_LENGTH = 32000

RULER_DATASETS = [
    # "qa_1",
    # "qa_2",
    # "vt",
    "fwe",
    # "niah_multikey_2",
    # "niah_multikey_3",
]
LONGBENCH_DATASETS = [
    # 'narrativeqa', 
    'qasper', 
    # 'multifieldqa_en', 
    # 'multifieldqa_zh', 
    # 'hotpotqa', 
    # '2wikimqa', 
    # 'musique', 
    # 'dureader', 
    # 'gov_report', 
    # 'qmsum', 
    # 'multi_news', 
    # 'vcsum', 
    # 'trec', 
    # 'triviaqa', 
    # 'samsum', 
    # 'lsht', 
    # 'passage_count', 
    # 'passage_retrieval_en', 
    # 'passage_retrieval_zh', 
    # 'lcc', 
    # 'repobench-p', 
    # 'qasper_e', 
    # 'multifieldqa_en_e', 
    # 'hotpotqa_e', 
    # '2wikimqa_e', 
    # 'gov_report_e', 
    # 'multi_news_e', 
    # 'trec_e', 
    # 'triviaqa_e', 
    # 'samsum_e', 
    # 'passage_count_e', 
    # 'passage_retrieval_en_e', 
    # 'lcc_e', 
    # 'repobench-p_e'
]

K_list = [8]
L_list = [60]
TAU_list = [0.2, 0.3, 0.5, 0.7]
heavy_list = [0.2]


def append_result_line(K, L, heavy_size, tau, dataset_name, score_value):
    line = f"K={K} L={L} tau={tau} heavy={heavy_size} | {dataset_name}={score_value}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def read_task_score(dataset_name: str) -> str:
    """Robust parsing: task_scores[dataset] can be dict or scalar (works for RULER + LongBench)."""
    if not METRICS_PATH.exists():
        return "NaN"
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        task_scores = metrics.get("task_scores", {})
        ts = task_scores.get(dataset_name, None)

        if isinstance(ts, dict) and ts:
            return str(next(iter(ts.values())))
        if isinstance(ts, (int, float)):
            return str(ts)
        return "NaN"
    except Exception:
        return "NaN"


def run_single(dataset_name: str, K: int, L: int, heavy_size: float, tau: float):
    adapter = None
    benchmark = None
    try:
        print(
            f"\n=== Dataset: {dataset_name} | "
            f"K={K}, L={L}, tau={tau}, heavy={heavy_size} | "
            f"BENCH={BENCHMARK_KIND} ==="
        )

        sparse_attention_config = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                SocketMaskerConfig(K=K, L=L, tau=tau, heavy_size=heavy_size),
            ]
        )

        adapter = ModelAdapterHF(
            model_name=MODEL_NAME,
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.bfloat16},
            generate_kwargs={"max_new_tokens": 32}
        )

        if BENCHMARK_KIND == "ruler":
            benchmark = Ruler32K([dataset_name])
        elif BENCHMARK_KIND == "longbench":
            benchmark = LongBench([dataset_name])
        elif BENCHMARK_KIND == "loogle":
            benchmark = Loogle(subsets_to_run=["shortdep_qa"])
        else:
            raise ValueError(
                f"BENCHMARK_KIND must be 'ruler' or 'longbench', got {BENCHMARK_KIND!r}"
            )

        benchmark.run_benchmark(
            adapter,
            RESULTS_ROOT,
            request_kwargs={"max_requests": MAX_REQUESTS, "max_context_length": MAX_CONTEXT_LENGTH},
        )

        score_value = read_task_score(dataset_name)
        append_result_line(K, L, heavy_size, tau, dataset_name, score_value)

    finally:
        del benchmark
        del adapter
        gc.collect()
        torch.cuda.empty_cache()


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    DATASETS = RULER_DATASETS if BENCHMARK_KIND == "ruler" else LONGBENCH_DATASETS

    for ds in DATASETS:
        if not ds:
            continue

        print("\n\n============================")
        print(f"### DATASET: {ds}")
        print("============================")

        for K, L, heavy_size, tau in itertools.product(K_list, L_list, heavy_list, TAU_list):
            run_single(ds, K, L, heavy_size, tau)


if __name__ == "__main__":
    main()

"""LOFT RAG benchmark implementation for long-context retrieval-augmented generation.

This benchmark implements LOFT's RAG evaluation exactly as specified in the LOFT paper,
ensuring 100% fidelity with LOFT's evaluation methodology and metrics.
"""

from typing import Any, Dict, List

import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("loft_rag")
class LoftRag(Benchmark):
    """LOFT RAG benchmark for evaluating long-context retrieval-augmented generation.

    LOFT (Long-context Open Foundation Tasks) RAG evaluates the ability of models to
    answer questions given long retrieved contexts. This benchmark includes:

    - Single-value RAG datasets: nq, hotpotqa, musique
    - Multi-value RAG datasets: qampari, quest

    Each dataset is available in multiple context lengths: 32k, 128k, 1m.

    Metrics:
    - Single-value: EM (Exact Match), Subspan EM, F1
    - Multi-value: EM, Coverage, Subspan EM

    Example:
        >>> loft_rag = LoftRag(subsets_to_run=["nq_32k", "hotpotqa_128k"])
        >>> results = loft_rag.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"EM score: {results['nq_32k']['em']}")
    """

    all_datasets: List[str] = [
        "nq_32k",
        "nq_128k",
        "nq_1m",
        "hotpotqa_32k",
        "hotpotqa_128k",
        "hotpotqa_1m",
        "musique_32k",
        "musique_128k",
        "musique_1m",
        "qampari_32k",
        "qampari_128k",
        "qampari_1m",
        "quest_32k",
        "quest_128k",
        "quest_1m",
    ]

    benchmark_name: str = "loft_rag"
    huggingface_dataset_id: str = "f20180301/rag"

    def _load_datasets(self) -> pd.DataFrame:
        """Load LOFT RAG datasets from HuggingFace Hub.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LOFT RAG datasets: {self.subsets_to_run}")
        dfs: List[pd.DataFrame] = []

        for subset in self.subsets_to_run:
            parts: List[str] = subset.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid subset format: {subset} (expected: dataset_length)"
                )

            length: str = parts[-1]
            dataset: str = "_".join(parts[:-1])
            hf_dataset_id: str = f"f20180301/loft-rag-{dataset}-{length}"

            from datasets import load_dataset

            dataset_dict = load_dataset(hf_dataset_id)

            subset_dfs: List[pd.DataFrame] = []
            for split_name in ["dev", "test"]:
                if split_name in dataset_dict:
                    split_df: pd.DataFrame = dataset_dict[split_name].to_pandas()
                    split_df["split"] = split_name
                    subset_dfs.append(split_df)

            if not subset_dfs:
                raise ValueError(f"No splits found for {subset} ({hf_dataset_id})")

            subset_df: pd.DataFrame = pd.concat(subset_dfs, ignore_index=True)
            subset_df["task"] = subset
            dfs.append(subset_df)
            print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")

        if not dfs:
            raise ValueError("No LOFT RAG subsets could be loaded")

        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")

        required_columns: List[str] = [
            "context",
            "question",
            "answers",
            "task",
            "answer_prefix",
            "max_new_tokens",
        ]
        missing_columns: List[str] = [
            col for col in required_columns if col not in combined_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LOFT RAG results.

        Args:
            results_df: DataFrame containing benchmark results

        Returns:
            Dictionary containing computed metrics
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        task_groups = results_df.groupby("task")
        task_metrics: Dict[str, Dict[str, float]] = {}
        all_em_scores: List[float] = []
        all_subspan_em_scores: List[float] = []
        all_f1_scores: List[float] = []
        all_coverage_scores: List[float] = []

        for task_name, task_df in task_groups:
            metrics: Dict[str, Any] = calculate_metrics(task_df)

            if "error" in metrics:
                print(f"  ❌ Error evaluating {task_name}: {metrics['error']}")
                continue

            task_metrics[task_name] = metrics
            all_em_scores.append(metrics["em"])
            all_subspan_em_scores.append(metrics["subspan_em"])

            if "f1" in metrics:
                all_f1_scores.append(metrics["f1"])
            if "coverage" in metrics:
                all_coverage_scores.append(metrics["coverage"])

            metric_str: str = (
                f"EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}"
            )
            if "f1" in metrics:
                metric_str += f", F1={metrics['f1']:.4f}"
            if "coverage" in metrics:
                metric_str += f", Coverage={metrics['coverage']:.4f}"
            print(f"  ✓ {task_name}: {metric_str}")

        overall_metrics: Dict[str, Any] = {
            "overall": {
                "em": (
                    float(sum(all_em_scores) / len(all_em_scores))
                    if all_em_scores
                    else 0.0
                ),
                "subspan_em": (
                    float(sum(all_subspan_em_scores) / len(all_subspan_em_scores))
                    if all_subspan_em_scores
                    else 0.0
                ),
            },
            "task_metrics": {
                task: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                for task, m in task_metrics.items()
            },
            "summary": {"total_tasks": len(task_metrics), "total_samples": len(results_df)},
        }

        if all_f1_scores:
            overall_metrics["overall"]["f1"] = float(
                sum(all_f1_scores) / len(all_f1_scores)
            )
        if all_coverage_scores:
            overall_metrics["overall"]["coverage"] = float(
                sum(all_coverage_scores) / len(all_coverage_scores)
            )

        overall_metrics["overall"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in overall_metrics["overall"].items()
        }

        return overall_metrics


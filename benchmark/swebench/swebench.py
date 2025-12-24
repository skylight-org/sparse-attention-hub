"""SWE-bench benchmark (API-based evaluation)."""

from typing import Any, Dict
import pandas as pd

from benchmark.base import Benchmark
from benchmark.benchmark_registry import register_benchmark


@register_benchmark("swebench")
class SWEBench(Benchmark):
    benchmark_name: str = "swebench"
    huggingface_dataset_id: str = "princeton-nlp/SWE-bench"

    def _load_datasets(self) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset(self.huggingface_dataset_id, split="test")
        df = ds.to_pandas()

        # Normalize to Sparse-Attention-Hub schema
        df["context"] = df["problem_statement"]
        df["question"] = "Produce a unified diff patch that fixes the issue."
        df["answer_prefix"] = ""
        df["task"] = "swebench"
        df["max_new_tokens"] = 2048
        df["context_length"] = None  # filled by executor

        return df

    def post_run_submit(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare SWE-bench submissions."""
        submissions = []

        for _, row in results_df.iterrows():
            # Ensure we don't submit empty patches if the model failed
            patch = row.get("predicted_answer", "")
            
            submissions.append({
                "run_id": str(row.get("run_id", "default")),
                "model_name": row.get("model_name"),
                "sparse_attention_config": row.get("sparse_attention_config"),
                "instance_id": row.get("instance_id"), # Crucial for evaluation
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "patch": patch,
                "metadata": {
                    "tokens_used": row.get("tokens_used"),
                    "latency_ms": row.get("latency_ms"),
                }
            })

        return {
            "benchmark": self.benchmark_name,
            "num_submissions": len(submissions),
            "submissions": submissions,
        }
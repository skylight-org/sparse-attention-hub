"""Benchmark helper for executing individual benchmark runs during config search."""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

import torch

from benchmark.executor_config import AdapterConfig
from benchmark.benchmark_registry import create_benchmark_instance
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from config_builders.utility import OBJECTIVE_FUNCTIONS
from OPTIMIZATION_EXPERIMENT import DRY_RUN
import random


class BenchmarkHelper:
    """Handles individual benchmark runs during config search.
    
    This class is responsible for executing a single benchmark run with a given
    sparse attention configuration and returning the evaluation metrics (score, density, error).
    """
    
    def __init__(self, 
            base_result_dir: Path,
            generation_kwargs: Dict[str, any],
            request_kwargs: Dict[str, any]) -> None:
        """Initialize the benchmark helper with configuration.
        
        Args:
            config: Dictionary containing benchmark configuration including:
                - search_result_dir: Base directory for search results
                - search_max_new_tokens: Maximum new tokens for generation
                - search_max_context_length: Maximum context length
                - search_max_requests: Maximum requests per trial
                - objective_function: Name of objective function to use
        """
        self.base_result_dir: Path = base_result_dir
        self.adapter_config: AdapterConfig = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.generation_kwargs: Dict[str, any] = generation_kwargs
        self.request_kwargs: Dict[str, any] = request_kwargs

    def __call__(self, attention_config: any, task_name: str, model_name: str) -> Tuple[float, float, float]:
        """Run benchmark and return (score, density, error) tuple.
        
        Args:
            attention_config: Sparse attention configuration to test
            task_name: Name of the benchmark task (may include subset, e.g., "benchmark/subset")
            model_name: Name of the model to use
            
        Returns:
            Tuple of (score, density, error) where:
                - score: Combined objective score (lower is better)
                - density: Attention density (0.0 to 1.0)
                - error: Attention output error (0.0 to 1.0)
        """
        try:
            # Early validation check - skip expensive benchmark if constraint fails
            if hasattr(attention_config, 'validity_constraint') and attention_config.validity_constraint is not None:
                if not attention_config.validity_constraint(attention_config):
                    logging.info(f"Config failed validity constraint, returning penalty score")
                    return 100.0, 1.0, 1.0  # Penalty score, worst density, worst error
            else:
                raise ValueError(f"No validity constraint found for attention configuration: {attention_config}. If there is no validity constraint . just set lambda: True in builder.")

            if hasattr(attention_config, 'objective') and attention_config.objective is not None:
                objective_function = OBJECTIVE_FUNCTIONS[attention_config.objective]
                logging.info(f"Using objective function: {objective_function.__name__} for attention configuration: {attention_config}")
            else:
                raise ValueError(f"No objective function found for attention configuration: {attention_config}. If config is objective agnostic just set default in builder.")
            
            if DRY_RUN:
                return random.random(), random.random(), random.random()

            benchmark_name: str
            subset_name: str | None
            benchmark_name, subset_name = task_name.split("/", 1) if "/" in task_name else (task_name, None)
            
            # Create result directory for this specific run
            result_dir: Path = os.path.join(self.base_result_dir, f"{model_name}_{task_name}_{hash(str(attention_config)) % 1000000}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Create model adapter
            adapter: ModelAdapterHF = ModelAdapterHF(
                model_name=model_name,
                sparse_attention_config=attention_config,
                model_kwargs=self.adapter_config.model_kwargs,
                tokenizer_kwargs=self.adapter_config.tokenizer_kwargs
            )
            
            # Create benchmark instance
            benchmark = create_benchmark_instance(
                benchmark_name=benchmark_name,
                subsets=[subset_name] if subset_name else None
            )
            print("The result directory is ", result_dir, flush=True)
            # Setup micro metric logger
            metric_logger: MicroMetricLogger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=str(result_dir),
                enabled_metrics=["research_attention_density", "research_attention_output_error"],
            )
            
            # Run benchmark directly
            metrics = benchmark.run_benchmark(
                adapter=adapter,
                result_dir=str(result_dir),
                generation_kwargs=self.generation_kwargs,
                request_kwargs=self.request_kwargs
            )
            
            # Flush the metric logger to ensure all metrics are written
            metric_logger.flush()
            
            # Extract micro metrics for sparse attention evaluation
            micro_metrics: Dict[str, float] = self._extract_micro_metrics(result_dir)
            error: float = micro_metrics["attention_error"]
            density: float = micro_metrics["density"]
            
            # For dense configuration (density=1.0, error=0.0), use a simple score
            if density == 1.0 and error == 0.0:
                # Dense baseline: use benchmark accuracy metrics instead of sparse metrics
                score: float = 100.0  # Small baseline score for dense
            else:
                # Use the selected objective function
                score = objective_function(error, density)
                # Also print to stdout so the test script can detect it
                print(f"Objective: {objective_function.__name__}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
                logging.info(f"Objective: {objective_function.__name__}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
            
            return score, density, error
                    
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
        return 5.0, 1.0, 1.0  # Penalty score, worst-case density, and worst-case error
    
    def _extract_micro_metrics(self, result_dir: Path) -> Dict[str, float]:
        """Extract attention error and density from micro metrics.
        
        Args:
            result_dir: Directory containing the micro_metrics.jsonl file
            
        Returns:
            Dictionary with keys:
                - attention_error: Average attention output error (0.0 to 1.0)
                - density: Average attention density (0.0 to 1.0)
        """
        micro_metrics_file: Path = os.path.join(result_dir, "micro_metrics.jsonl")
        if not os.path.exists(micro_metrics_file):
            # For dense configuration, micro_metrics.jsonl won't exist since no sparse attention is used
            # Return default values: 0 error (perfect) and 1.0 density (fully dense)
            logging.info(f"micro_metrics.jsonl not found in {result_dir}, using dense defaults")
            return {"attention_error": 0.0, "density": 1.0}
            
        errors: list[float] = []
        densities: list[float] = []
        with open(micro_metrics_file, "r") as f:
            for line in f:
                try:
                    entry: dict = json.loads(line.strip())
                    metric: str | None = entry.get("metric")
                    value: any = entry.get("value")
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        if metric == "research_attention_output_error": 
                            errors.append(float(value))
                        elif metric == "research_attention_density": 
                            densities.append(float(value))
                except (json.JSONDecodeError, ValueError, TypeError): 
                    continue
                    
        return {
            "attention_error": sum(errors) / len(errors) if errors else 1.0, 
            "density": sum(densities) / len(densities) if densities else 1.0
        }


#!/usr/bin/env python3
"""
Hyperparameter search for optimal sparse attention configurations.
"""

import fire
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

import torch
import ray
from ray import tune

from benchmark.executor_config import AdapterConfig, BenchmarkConfig
from benchmark.benchmark_registry import create_benchmark_instance
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from optimizer_factory import create_optimizer
from config_builders.utility import (
    get_masker_list_name, 
    create_sparsity_objective, 
    OBJECTIVE_FUNCTIONS,
    OptimalConfig,
    get_all_masker_config_classes,
    serialize_sparse_config,
    deserialize_sparse_config,
)
from config_builders.factory import build_all_configs

# Import all masker configs
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig


class BenchmarkHelper:
    """Handles individual benchmark runs during config search."""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_result_dir = Path(config["search_result_dir"])
        self.adapter_config = AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.generation_kwargs = {
            "max_new_tokens": config["search_max_new_tokens"],
            "do_sample": False
        }
        self.request_kwargs = {
            "max_context_length": config["search_max_context_length"],
            "max_requests": config["search_max_requests"],
        }
        
        # Get objective function
        self.objective_name = config.get("objective_function", "default")
        self.objective_function = OBJECTIVE_FUNCTIONS.get(self.objective_name, OBJECTIVE_FUNCTIONS["default"])
        logging.info(f"Using objective function: {self.objective_name}")

    def __call__(self, attention_config, task_name: str, model_name: str) -> Tuple[float, float, float]:
        """Run benchmark and return (score, density, error) tuple."""
        try:
            # Early validation check - skip expensive benchmark if constraint fails
            if hasattr(attention_config, 'validity_constraint') and attention_config.validity_constraint is not None:
                if not attention_config.validity_constraint(attention_config):
                    logging.info(f"Config failed validity constraint, returning penalty score")
                    return 100.0, 1.0, 1.0  # Penalty score, worst density, worst error
            
            benchmark_name, subset_name = task_name.split("/", 1) if "/" in task_name else (task_name, None)
            
            # Create result directory for this specific run
            result_dir = self.base_result_dir / f"{model_name}_{task_name}_{hash(str(attention_config)) % 1000000}"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Create model adapter
            adapter = ModelAdapterHF(
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
            metric_logger = MicroMetricLogger()
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
            micro_metrics = self._extract_micro_metrics(result_dir)
            error, density = micro_metrics["attention_error"], micro_metrics["density"]
            
            # For dense configuration (density=1.0, error=0.0), use a simple score
            if density == 1.0 and error == 0.0:
                # Dense baseline: use benchmark accuracy metrics instead of sparse metrics
                score = 100.0  # Small baseline score for dense
            else:
                # Use the selected objective function
                score = self.objective_function(error, density)
                # Also print to stdout so the test script can detect it
                print(f"Objective: {self.objective_name}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
                logging.info(f"Objective: {self.objective_name}, Error: {error:.4f}, Density: {density:.4f}, Score: {score:.4f}")
            
            return score, density, error
                    
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
        return 5.0, 1.0, 1.0  # Penalty score, worst-case density, and worst-case error
    
    def _extract_micro_metrics(self, result_dir: Path) -> dict:
        """Extract attention error and density from micro metrics."""
        micro_metrics_file = result_dir / "micro_metrics.jsonl"
        if not micro_metrics_file.exists():
            # For dense configuration, micro_metrics.jsonl won't exist since no sparse attention is used
            # Return default values: 0 error (perfect) and 1.0 density (fully dense)
            logging.info(f"micro_metrics.jsonl not found in {result_dir}, using dense defaults")
            return {"attention_error": 0.0, "density": 1.0}
            
        errors, densities = [], []
        with open(micro_metrics_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    metric, value = entry.get("metric"), entry.get("value")
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

class ConfigSearchManager:
    """Manages Phase 1: Hyperparameter search for optimal configs."""
    
    def __init__(self, base_config: dict):
        self.config = base_config
        # Add timestamp to the results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(base_config["optimal_configs_dir"])
        self.results_dir = base_dir / f"run_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        print(f"Saving optimal configs to: {self.results_dir}")
        
    def search_optimal_config(
        self, 
        model: str, 
        task: str, 
        masker_name: str, 
        masker_classes: Optional[List],
        full_sparse_config: Optional[ResearchAttentionConfig] = None,
        actors_per_gpu: int = 1
    ) -> OptimalConfig:
        """Search for optimal hyperparameters for a single combination."""
        
        config_file = self.results_dir / f"{model}_{task}_{masker_name}.json".replace("/", "_")
        
        # Check if already exists
        if config_file.exists() and not self.config.get("force_search", False):
            print(f"  → Loading existing config")
            return self._load_config(config_file)
        
        # Handle dense config (no optimization needed)
        if masker_classes is None:
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=None,
                masker_classes=None,
                hyperparams={},
                score=0.0,
                search_time=0.0,
                num_trials=1
            )
            self._save_config(optimal, config_file)
            return optimal
        
        # Run hyperparameter search
        start_time = time.time()
        
        try:
            # Create optimizer with template config for fixed parameters
            optimizer = create_optimizer(full_sparse_config)
            
            # Show what we're searching
            search_space = optimizer.create_search_space(task)
            print(f"  → Search space parameters:")
            for param, space_obj in search_space.items():
                # Extract actual values from Ray Tune objects
                if hasattr(space_obj, 'categories'):
                    values = space_obj.categories
                    print(f"     - {param}: {values}")
                else:
                    print(f"     - {param}: {space_obj}")
            
            # Create objective function
            def objective(trial_config):
                runner = BenchmarkHelper(self.config)
                attention_config = optimizer.create_config_from_params(trial_config)
                score, density, error = runner(attention_config, task, model)
                return {"combined_score": score, "density": density, "error": error}
            
            # ### run a sample objective to ensure there are no errors
            print("="*10, "Running a short test objective to ensure there are no errors", flush=True)
            sample_config = {
                "AdaptiveSamplingMaskerConfig_base_rate_sampling": 0.1,
                "AdaptiveSamplingMaskerConfig_epsilon": 0.25,
                "AdaptiveSamplingMaskerConfig_delta": 0.25
            }
            result = objective(sample_config)
            print("="*10, "Successfully ran a short test objective", flush=True)
            print(sample_config)
            print(result)
            print("="*100, flush=True)
            
            # Run Ray Tune
            sanitized_name = f"{model}_{task}_{masker_name}".replace("/", "_")
            analysis = tune.run(
                objective,
                config=search_space,
                metric="combined_score",
                mode="min",
                resources_per_trial={"CPU": 1, "GPU": 1.0 / actors_per_gpu},
                storage_path=os.path.abspath(self.config["ray_results_dir"]),
                name=sanitized_name,
                verbose=1,  # Show Ray Tune progress
                stop={"training_iteration": 1},  # One evaluation per config
            )
            
            # Get best config
            best_trial = analysis.get_best_trial("combined_score", "min", "last")
            best_config = optimizer.create_config_from_params(best_trial.config)
            
            # Save detailed trial information for post-analysis
            trials_info = []
            for trial in analysis.trials:
                trial_info = {
                    "trial_id": trial.trial_id,
                    "config": trial.config,
                    "score": trial.last_result.get("combined_score", float('inf')) if trial.last_result else float('inf'),
                    "status": trial.status,
                    "start_time": trial.start_time.isoformat() if hasattr(trial, 'start_time') and trial.start_time else None,
                    "metric_history": trial.metric_analysis.get("combined_score", {}) if hasattr(trial, 'metric_analysis') else {}
                }
                trials_info.append(trial_info)
            
            # Save trial details to separate file
            trials_file = self.results_dir / f"{model}_{task}_{masker_name}_trials.json".replace("/", "_")
            with open(trials_file, "w") as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "masker_name": masker_name,
                    "objective_function": self.config.get("objective_function", "default"),
                    "best_trial_id": best_trial.trial_id,
                    "trials": trials_info,
                    "analysis_dataframe_path": str(self.results_dir / f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_"))
                }, f, indent=2)
            
            # Save Ray analysis dataframe for detailed analysis
            df = analysis.dataframe()
            df.to_csv(self.results_dir / f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_"), index=False)
            
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=best_config,
                masker_classes=masker_classes,
                hyperparams=best_trial.config,
                score=best_trial.last_result["combined_score"],
                search_time=time.time() - start_time,
                num_trials=len(analysis.trials)
            )
            
            self._save_config(optimal, config_file)
            return optimal
            
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
            traceback.print_exc()
            # Return failure config
            optimal = OptimalConfig(
                model=model,
                task=task,
                masker_name=masker_name,
                sparse_config=full_sparse_config,
                masker_classes=masker_classes,
                hyperparams={},
                score=5.0,
                search_time=time.time() - start_time,
                num_trials=0
            )
            self._save_config(optimal, config_file)
            return optimal
    
    def _save_config(self, config: OptimalConfig, filepath: Path):
        """Save configuration to JSON."""
        data = asdict(config)
        
        # Convert sparse config to serializable format
        if config.sparse_config:
            data["sparse_config"] = serialize_sparse_config(config.sparse_config)
        
        # Convert masker classes to strings
        if config.masker_classes:
            data["masker_classes"] = [cls.__name__ for cls in config.masker_classes]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_config(self, filepath: Path) -> OptimalConfig:
        """Load configuration from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct sparse config if present
        if data.get("sparse_config"):
            data["sparse_config"] = deserialize_sparse_config(data["sparse_config"])
        
        # Reconstruct masker classes from strings
        if data.get("masker_classes"):
            # Dynamically discover all available masker config classes
            class_map = get_all_masker_config_classes()
            data["masker_classes"] = [class_map[name] for name in data["masker_classes"]]
        
        return OptimalConfig(**data)
    
def run_search(config: dict, actors_per_gpu: int = 1) -> Dict[str, OptimalConfig]:
    """Find optimal configurations for all combinations."""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Models: {len(config['models'])}")
    print(f"Tasks: {len(config['tasks'])}")
    print(f"Optimal Configs: {len(config['optimal_configs'])}")
    print(f"To Optimize Configs: {len(config['to_optimize_configs'])}")
    print(f"Total Combinations to optimize: {len(config['models']) * len(config['tasks']) * len(config['to_optimize_configs'])}")
    print(f"Samples per search: {config['num_samples']}")
    print(f"Objective Function: {config['objective_function']}")
    
    # Display objective function details
    if config['objective_function'].startswith('sparsity_'):
        target = int(config['objective_function'].split('_')[1])
        print(f"  → Targeting {target}% density (0.{target:02d} fraction)")
        print(f"  → Formula: 0.99 * error + 0.01 * density + penalty for exceeding target")
    
    print("\nSearch Configuration:")
    print(f"  → Max new tokens: {config['search_max_new_tokens']}")
    print(f"  → Max context length: {config['search_max_context_length']}")
    print(f"  → Max requests per trial: {config['search_max_requests']}")
    print(f"  → Timeout per trial: {config['search_timeout']}s")
    
    print("\nNote: For each sparse config, Ray Tune will search different hyperparameter")
    print("values (e.g., window_size, sink_size, sampling_rate) to find the best combination.")
    print("="*80)
    
    manager = ConfigSearchManager(config)
    optimal_configs = {}
    
    total = len(config["models"]) * len(config["tasks"]) * len(config["to_optimize_configs"]) + len(config["models"]) * len(config["tasks"]) * len(config["optimal_configs"])
    current = 0
    
    for model in config["models"]:
        print(f"\nModel: {model}")
        print("-" * 60)
        
        for task in config["tasks"]:
            for masker_name, (masker_classes, full_config) in config["to_optimize_configs_map"].items():
                current += 1
                key = f"{model}_{task}_{masker_name}".replace("/", "_")
                
                print(f"\n[{current}/{total}] Task: {task} | Config: {masker_name}")
                optimal = manager.search_optimal_config(
                    model, task, masker_name, masker_classes, full_config, actors_per_gpu
                )
                optimal_configs[key] = optimal
            
            for masker_name, (masker_classes, full_config) in config["optimal_configs_map"].items():
                current += 1
                key = f"{model}_{task}_{masker_name}".replace("/", "_")
                optimal = OptimalConfig(
                    model=model,
                    task=task,
                    masker_name=masker_name,
                    sparse_config=full_config,
                    masker_classes=masker_classes,
                    hyperparams={},
                    score=0.0,
                    search_time=0.0,
                    num_trials=0
                )
                manager._save_config(optimal, Path(manager.results_dir) / f"{key}.json")
                optimal_configs[key] = optimal
    
    print(f"\n{'='*80}")
    print(f"Search complete. Found {len(optimal_configs)} optimal configurations.")
    print(f"Configs saved to: {manager.results_dir}")
    print(f"Run identifier: {manager.timestamp}")
    print(f"{'='*80}")
    
    return optimal_configs

################################################################# CONFIGURE YOUR RUN HERE #################################################################

# Model configurations
# Weight files are loaded from SPARSE_ATTENTION_WEIGHTS_DIR environment variable
# Set it to the directory containing your HashAttention weight files
weights_dir = os.environ.get("SPARSE_ATTENTION_WEIGHTS_DIR", "./weights")
MODEL_CONFIGS = {
    "llama": {
        "weight_file": os.path.join(weights_dir, "llama3.1-8b-patch.64K.v1.hat_weights.pkl"),
        "model_name": "meta-llama/Llama-3.1-8B-Instruct"
    },
    "deepseek": {
        "weight_file": os.path.join(weights_dir, "DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K_hat_weights.pkl"),
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    },
    "mistral": {
        "weight_file": os.path.join(weights_dir, "Mistral-7B-Instruct-v0.3.24K.20.500.hat_weights.pkl"),
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3"
    }
}

DEFAULT_MODEL = "llama"

# Task configurations
DEBUG_TASKS = ["loogle/shortdep_qa"]

RUN_TASKS = [
    "ruler32k/vt",
     "ruler32k/qa_1",
     "ruler32k/qa_2",
     "ruler32k/fwe",
     "ruler32k/niah_multikey_2",
     "ruler32k/niah_multikey_3",
]

def get_all_sparse_configs(weight_file: str = None, objective: str = "default", memory_objective: str = None) -> List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]:
    """Get all sparse attention configurations.
    Returns list of (name, full_config, masker_classes) tuples.
    
    Note: The configs returned here are only used to determine which masker classes
    to use. The actual parameter values will be determined by Ray Tune search.
    
    Args:
        weight_file: Path to weight file (required)
        objective: Objective function name (e.g., "sparsity_5")
        memory_objective: Memory objective parameter for configs that need it
        
    Returns:
        Tuple of (optimal_configs, to_optimize_configs)
    """
    assert weight_file is not None, "Weight file is required for HashAttention Masker"
    
    # Use factory to build all configs
    # Currently using double_sparsity builder, can be extended to use multiple builders
    optimal_configs, to_optimize_configs = build_all_configs(
        weight_file=weight_file,
        objective=objective,
        builder_names=["magicpig"],  # Specify which builders to use
        memory_objective=memory_objective
    )
    
    return optimal_configs, to_optimize_configs


def get_run_configuration(
    objective: str,
    debug: bool,
    num_samples: int,
    search_timeout: int,
    search_max_new_tokens: int,
    search_max_context_length: int,
    search_max_requests: int,
    force_search: bool,
    optimal_configs_dir: str,
    ray_results_dir: str,
    memory_objective: str = None
) -> dict:
    """Build complete configuration from command-line arguments."""
    num_gpus = torch.cuda.device_count()
    
    # Get model configuration  
    model_config = MODEL_CONFIGS[DEFAULT_MODEL]
    weight_file = model_config["weight_file"]
    model_name = model_config["model_name"]

    if not os.path.exists(weight_file):
        weight_file = "./hat_weights.pkl"
        print(f"Warning: HashAttention weights not found, using {weight_file}")
    
    # Get all sparse configs
    optimal_configs, to_optimize_configs = get_all_sparse_configs(weight_file, objective=objective, memory_objective=memory_objective)
    
    # Filter configs based on debug mode
    if debug:
        sparse_configs = to_optimize_configs[:3]  # Just first 3 for debug
        models = [model_name]
        tasks = DEBUG_TASKS
        num_samples = 8
    else:
        models = [model_name]
        tasks = RUN_TASKS
        # num_samples is already passed as parameter
    
    # Build config maps
    optimal_configs_map = {}
    to_optimize_configs_map = {}
    for name, full_config, classes in optimal_configs:
        optimal_configs_map[name] = (classes, full_config)
    for name, full_config, classes in to_optimize_configs:
        to_optimize_configs_map[name] = (classes, full_config)
    
    return {
        "models": models,
        "tasks": tasks,
        "optimal_configs": optimal_configs,
        "to_optimize_configs": to_optimize_configs,
        "optimal_configs_map": optimal_configs_map,
        "to_optimize_configs_map": to_optimize_configs_map,
        "gpu_ids": list(range(num_gpus)),
        "num_samples": num_samples,
        "objective_function": objective,
        
        # Directories
        "optimal_configs_dir": optimal_configs_dir,
        "ray_results_dir": ray_results_dir,
        "search_result_dir": os.path.join(ray_results_dir, "search_runs"),
        
        # Search params
        "search_timeout": search_timeout,
        "search_max_new_tokens": search_max_new_tokens,
        "search_max_context_length": search_max_context_length,
        "search_max_requests": search_max_requests,
        "force_search": force_search,
    }

######################################################### CONFIGURATION ENDS HERE #########################################################``

def main(
    objective: str,
    num_samples: int,
    search_max_new_tokens: int,
    search_max_context_length: int,
    search_max_requests: int,
    debug: bool = False,
    force_search: bool = False,
    optimal_configs_dir: str = "./optimal_configs",
    ray_results_dir: str = "./ray_results",
    search_timeout: int = 900,
    actors_per_gpu: int = 1,
    memory_objective: str = None,
):
    """
    Hyperparameter search for sparse attention methods.
    
    Args:
        objective: Objective function to use for optimization (required)
        num_samples: Number of samples per hyperparameter search (required)
        search_max_new_tokens: Max new tokens for search trials (required)
        search_max_context_length: Max context length for search trials (required)
        search_max_requests: Max requests per search trial (required)
        debug: Debug mode with minimal configs (default: False)
        force_search: Force re-run of search even if configs exist (default: False)
        optimal_configs_dir: Directory for storing optimal configurations (default: "./optimal_configs")
        ray_results_dir: Directory for Ray Tune results (default: "./ray_results")
        search_timeout: Timeout per search trial in seconds (default: 900)
        actors_per_gpu: Number of actors per GPU for resource allocation (default: 1)
        memory_objective: Memory objective parameter (e.g., "memory_32") for configs that need it (default: None)
    """
    # Validate objective function
    if objective not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Invalid objective function '{objective}'. Choose from: {list(OBJECTIVE_FUNCTIONS.keys())}")
    
    config = get_run_configuration(
        objective=objective,
        debug=debug,
        num_samples=num_samples,
        search_timeout=search_timeout,
        search_max_new_tokens=search_max_new_tokens,
        search_max_context_length=search_max_context_length,
        search_max_requests=search_max_requests,
        force_search=force_search,
        optimal_configs_dir=optimal_configs_dir,
        ray_results_dir=ray_results_dir,
        memory_objective=memory_objective,
    )
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, 
                runtime_env={"working_dir": str(root_path)})
    
    optimal_configs = run_search(config, actors_per_gpu)
    ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fire.Fire(main)

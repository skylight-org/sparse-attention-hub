#!/usr/bin/env python3
"""
Hyperparameter search for optimal sparse attention configurations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

import torch
import ray

from config_builders.utility import OBJECTIVE_FUNCTIONS, OptimalConfig
from config_builders.factory import build_all_configs

# Import all masker configs
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

# Import search manager
from search_manager import ConfigSearchManager

# Import run configuration
from OPTIMIZATION_EXPERIMENT import (
    MODEL_CONFIGS,
    MODELS,
    TASKS,
    SPARSITY_OBJECTIVES,
    MEMORY_OBJECTIVES,
    SEARCH_MAX_NEW_TOKENS,
    SEARCH_MAX_CONTEXT_LENGTH,
    SEARCH_MAX_REQUESTS,
    FORCE_SEARCH,
    OPTIMAL_CONFIGS_DIR,
    RAY_RESULTS_DIR,
    ACTORS_PER_GPU,
    BUILDER_NAMES,
)

def get_all_sparse_configs(model_config: Dict[str, str], 
                          sparsity_objectives: List[int], 
                          memory_objectives: List[int], 
                          builder_names: List[str]) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]], 
                                                            List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
    """Get all sparse attention configurations.
    Returns list of (name, full_config, masker_classes) tuples.
    
    Note: The configs returned here are only used to determine which masker classes
    to use. The actual parameter values will be determined by Ray Tune search.
    
    Args:
        weight_file: Path to weight file (required)
        objective: Objective function name (e.g., "sparsity_5")
        memory_objective: Memory objective parameter for configs that need it
        builder_names: List of builder names to use
        
    Returns:
        Tuple of (optimal_configs, to_optimize_configs)
    """
    # Use factory to build all configs
    optimal_configs, to_optimize_configs = build_all_configs(
        model_config = model_config,
        sparsity_objectives=sparsity_objectives,
        memory_objectives=memory_objectives,
        builder_names=builder_names,
    )
    
    return optimal_configs, to_optimize_configs


def run_search() -> Dict[str, OptimalConfig]:
    """Find optimal configurations for all combinations.
    
    This function orchestrates the search process across all model/task/config
    combinations, using ConfigSearchManager to handle individual searches.
    All configuration is loaded from OPTIMIZATION_EXPERIMENT.py.
    
    Args:
        actors_per_gpu: Number of actors per GPU for resource allocation
        
    Returns:
        Dictionary mapping config keys to OptimalConfig objects
    """
    
    manager: ConfigSearchManager = ConfigSearchManager(
        optimal_configs_dir=OPTIMAL_CONFIGS_DIR,
        force_search=FORCE_SEARCH,
        generation_kwargs={
            "max_new_tokens": SEARCH_MAX_NEW_TOKENS,
            "do_sample": False
        },
        request_kwargs={
            "max_context_length": SEARCH_MAX_CONTEXT_LENGTH,
            "max_requests": SEARCH_MAX_REQUESTS
        },
        ray_results_dir=RAY_RESULTS_DIR
    )
    final_optimal_configs: Dict[str, OptimalConfig] = {}

    # first run all the optimal configs
    for model in MODELS:
        # Get model configuration  
        model_config: Dict[str, str] = MODEL_CONFIGS[model]
        
        # Get all sparse configs
        optimal_configs, to_optimize_configs = get_all_sparse_configs(
            model_config, 
            sparsity_objectives=SPARSITY_OBJECTIVES, 
            memory_objectives=MEMORY_OBJECTIVES,
            builder_names=BUILDER_NAMES
        )
        for task in TASKS:
            for (masker_name, full_config, masker_classes) in optimal_configs:
                key = f"{model}_{task}_{masker_name}".replace("/", "_")
                optimal = OptimalConfig(
                    model=model_config["model_name"],
                    task=task,
                    masker_name=masker_name,
                    sparse_config=full_config,
                    masker_classes=masker_classes,
                    hyperparams={},
                    score=0.0,
                    search_time=0.0,
                    num_trials=0
                )
                manager._save_config(optimal, os.path.join(manager.results_dir, f"{key}.json"))
                final_optimal_configs[key] = optimal
    
        
        for task in TASKS:
            for (masker_name, full_config, masker_classes) in to_optimize_configs:
                key: str = f"{model}_{task}_{masker_name}".replace("/", "_")
                
                optimal: OptimalConfig = manager.search_optimal_config(
                    model_config["model_name"], task, masker_name, masker_classes, full_config, ACTORS_PER_GPU
                )
                final_optimal_configs[key] = optimal

    return final_optimal_configs

def main() -> None:
    """Hyperparameter search for sparse attention methods.
    
    All configuration is loaded from OPTIMIZATION_EXPERIMENT.py. Modify that file to change
    search parameters instead of passing command-line arguments.
    """

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, 
                runtime_env={"working_dir": str(root_path)})
    
    run_search()
    ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()

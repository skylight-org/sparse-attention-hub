"""Search manager for orchestrating Ray Tune hyperparameter search."""

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{current_dir}:{root_path}"

from ray import tune

from optimizer_factory import create_optimizer
from config_builders.utility import (
    OptimalConfig,
    get_all_masker_config_classes,
    serialize_sparse_config,
    deserialize_sparse_config,
)
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from benchmark_helper import BenchmarkHelper
from OPTIMIZATION_EXPERIMENT import USE_TIMESTAMP_FOR_RESULTS_DIR

class ConfigSearchManager:
    """Manages Phase 1: Hyperparameter search for optimal configs.
    
    This class orchestrates Ray Tune hyperparameter search to find optimal
    sparse attention configurations for given model/task combinations.
    """
    
    def __init__(self, optimal_configs_dir: str, 
    force_search: bool, 
    generation_kwargs: Dict[str, any], 
    request_kwargs: Dict[str, any],
    ray_results_dir: str) -> None:
        """Initialize the search manager with configuration.
        
        Args:
            base_config: Dictionary containing search configuration including:
                - optimal_configs_dir: Directory to save optimal configs
                - force_search: Whether to force re-search even if configs exist
        """
        # Add timestamp to the results directory
        if USE_TIMESTAMP_FOR_RESULTS_DIR:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp: str = "default"
        self.results_dir: str = os.path.join(optimal_configs_dir, f"run_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        self.force_search: bool = force_search
        self.generation_kwargs: Dict[str, any] = generation_kwargs
        self.request_kwargs: Dict[str, any] = request_kwargs
        self.ray_results_dir: Path = ray_results_dir
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
        """Search for optimal hyperparameters for a single combination.
        
        Args:
            model: Model name to use
            task: Task name to benchmark
            masker_name: Name of the masker configuration
            masker_classes: List of masker classes (None for dense configs)
            full_sparse_config: Full sparse attention config template
            actors_per_gpu: Number of actors per GPU for resource allocation
            
        Returns:
            OptimalConfig containing the best configuration found
        """
        config_file: Path = os.path.join(self.results_dir, f"{model}_{task}_{masker_name}.json".replace("/", "_"))
        
        # Check if already exists
        if os.path.exists(config_file) and not self.force_search:
            print(f"  → Loading existing config")
            return self._load_config(config_file)
        
        # Handle dense config (no optimization needed)
        if masker_classes is None:
            optimal: OptimalConfig = OptimalConfig(
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
        start_time: float = time.time()
        
        try:
            # Create optimizer with template config for fixed parameters
            optimizer = create_optimizer(full_sparse_config)
            
            # Show what we're searching
            search_space: Dict[str, any] = optimizer.create_search_space(task)
            print(f"  → Search space parameters:")
            for param, space_obj in search_space.items():
                # Extract actual values from Ray Tune objects
                if hasattr(space_obj, 'categories'):
                    values = space_obj.categories
                    print(f"     - {param}: {values}")
                else:
                    print(f"     - {param}: {space_obj}")
            
            # Create objective function
            def objective(trial_config: Dict[str, any]) -> Dict[str, float]:
                runner: BenchmarkHelper = BenchmarkHelper(
                    base_result_dir=self.results_dir,
                    generation_kwargs=self.generation_kwargs,
                    request_kwargs=self.request_kwargs
                )
                attention_config = optimizer.create_config_from_params(trial_config)
                score: float
                density: float
                error: float
                score, density, error = runner(attention_config, task, model)
                return {"combined_score": score, "density": density, "error": error}
            
            # # ### run a sample objective to ensure there are no errors
            # print("="*10, "Running a short test objective to ensure there are no errors", flush=True)
            # sample_config: Dict[str, float] = {
            #     "AdaptiveSamplingMaskerConfig_base_rate_sampling": 0.1,
            #     "AdaptiveSamplingMaskerConfig_epsilon": 0.25,
            #     "AdaptiveSamplingMaskerConfig_delta": 0.25
            # }
            # result: Dict[str, float] = objective(sample_config)
            # print("="*10, "Successfully ran a short test objective", flush=True)
            # print(sample_config)
            # print(result)
            # print("="*100, flush=True)
            
            # Run Ray Tune
            sanitized_name: str = f"{model}_{task}_{masker_name}".replace("/", "_")
            analysis = tune.run(
                objective,
                config=search_space,
                metric="combined_score",
                mode="min",
                resources_per_trial={"CPU": 1, "GPU": 1.0 / actors_per_gpu},
                storage_path=self.ray_results_dir,
                name=sanitized_name,
                verbose=1,  # Show Ray Tune progress
                stop={"training_iteration": 1},  # One evaluation per config
            )
            
            # Get best config
            best_trial = analysis.get_best_trial("combined_score", "min", "last")
            best_config = optimizer.create_config_from_params(best_trial.config)
            
            # Save detailed trial information for post-analysis
            trials_info: List[Dict[str, any]] = []
            for trial in analysis.trials:
                trial_info: Dict[str, any] = {
                    "trial_id": trial.trial_id,
                    "config": trial.config,
                    "score": trial.last_result.get("combined_score", float('inf')) if trial.last_result else float('inf'),
                    "status": trial.status,
                    "start_time": trial.start_time.isoformat() if hasattr(trial, 'start_time') and trial.start_time else None,
                    "metric_history": trial.metric_analysis.get("combined_score", {}) if hasattr(trial, 'metric_analysis') else {}
                }
                trials_info.append(trial_info)
            
            # Save trial details to separate file
            trials_file: Path = os.path.join(self.results_dir, f"{model}_{task}_{masker_name}_trials.json".replace("/", "_"))
            with open(trials_file, "w") as f:
                json.dump({
                    "model": model,
                    "task": task,
                    "masker_name": masker_name,
                    "objective_function": full_sparse_config.objective if full_sparse_config.objective else "None",
                    "best_trial_id": best_trial.trial_id,
                    "trials": trials_info,
                    "analysis_dataframe_path": str(os.path.join(self.results_dir, f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_")))
                }, f, indent=2)
            
            # Save Ray analysis dataframe for detailed analysis
            df = analysis.dataframe()
            df.to_csv(os.path.join(self.results_dir, f"{model}_{task}_{masker_name}_analysis.csv".replace("/", "_")), index=False)
            
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
    
    def _save_config(self, config: OptimalConfig, filepath: Path) -> None:
        """Save configuration to JSON.
        
        Args:
            config: OptimalConfig to save
            filepath: Path where to save the config
        """
        data: Dict[str, any] = asdict(config)
        
        # Convert sparse config to serializable format
        if config.sparse_config:
            data["sparse_config"] = serialize_sparse_config(config.sparse_config)
        
        # Convert masker classes to strings
        if config.masker_classes:
            data["masker_classes"] = [cls.__name__ for cls in config.masker_classes]
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_config(self, filepath: Path) -> OptimalConfig:
        """Load configuration from JSON.
        
        Args:
            filepath: Path to the config file to load
            
        Returns:
            OptimalConfig loaded from file
        """
        with open(filepath, "r") as f:
            data: Dict[str, any] = json.load(f)
        
        # Reconstruct sparse config if present
        if data.get("sparse_config"):
            data["sparse_config"] = deserialize_sparse_config(data["sparse_config"])
        
        # Reconstruct masker classes from strings
        if data.get("masker_classes"):
            # Dynamically discover all available masker config classes
            class_map: Dict[str, type] = get_all_masker_config_classes()
            data["masker_classes"] = [class_map[name] for name in data["masker_classes"]]
        
        return OptimalConfig(**data)


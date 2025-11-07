#!/usr/bin/env python3
"""
Ray-based parallel benchmark runner with efficient resource management.

This implementation uses Ray for:
- Distributed execution with automatic resource management
- Efficient model caching through Ray actors
- Built-in fault tolerance and progress tracking
- Optimal task scheduling to minimize model loading

Usage:
    python benchmark/raytune/run_ray_benchmarks.py ./optimal_configs/run_20250818_203531
"""

import fire
import json
import logging
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import traceback

# Path setup
current_dir = Path(__file__).parent
root_path = current_dir.parent.parent
sys.path.extend([str(current_dir), str(root_path)])

import ray
from ray.util.queue import Queue as RayQueue
from ray.util.actor_pool import ActorPool

from benchmark.benchmark_registry import create_benchmark_instance
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from config_builders.utility import deserialize_sparse_config



@dataclass
class BenchmarkTask:
    """Single benchmark task to execute."""
    task_id: str
    model_name: str
    task_name: str
    masker_name: str
    sparse_config: Optional[Dict]  # JSON-serializable config
    result_dir: str
    generation_kwargs: Dict[str, Any]
    request_kwargs: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result from a benchmark execution."""
    task_id: str
    success: bool
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    model_load_time: float = 0.0


@ray.remote(num_gpus=1)
class GPUBenchmarkActor:
    """Ray actor that runs benchmarks on a specific GPU with fresh model initialization for each task."""
    
    def __init__(self, actor_id: int, adapter_config: Dict):
        self.actor_id = actor_id
        self.adapter_config = adapter_config
        
        # Ray sets CUDA_VISIBLE_DEVICES for us, so GPU 0 is always the correct device
        torch.cuda.set_device(0)
        
        # Get actual GPU info for logging
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Actor {actor_id} initialized on GPU {gpu_name}")
     
    def _create_fresh_model(self, model_name: str, sparse_config: Optional[Dict], 
                          masker_name: str, task_name: str) -> Tuple[ModelAdapterHF, float]:
        """Create a fresh model from scratch for each task.
        
        This ensures no state leakage between tasks with different sparse configs.
        Returns (model, load_time).
        """
        logging.info(f"Actor {self.actor_id}: Creating fresh model for {task_name} with {masker_name}")
        
        # Clear any GPU cache before loading
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # Reconstruct sparse config
        sparse_attention_config = deserialize_sparse_config(sparse_config)
        
        # Create completely fresh model instance
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config,
            model_kwargs=self.adapter_config["model_kwargs"],
            tokenizer_kwargs=self.adapter_config["tokenizer_kwargs"]
        )
        
        load_time = time.time() - start_time
        logging.info(f"Actor {self.actor_id}: Model created in {load_time:.1f}s")
        
        return adapter, load_time
    
    def run_benchmark(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute a single benchmark task."""
        total_start = time.time()
        
        adapter = None
        try:
            # Create fresh model for this task
            adapter, model_load_time = self._create_fresh_model(
                task.model_name, task.sparse_config, task.masker_name, task.task_name
            )
            
            # Parse benchmark info
            benchmark_name, subset = (task.task_name.split("/", 1) 
                                    if "/" in task.task_name 
                                    else (task.task_name, None))
            
            # Create benchmark instance
            benchmark = create_benchmark_instance(
                benchmark_name=benchmark_name,
                subsets=[subset] if subset else None
            )
            
            # Setup result directory
            Path(task.result_dir).mkdir(parents=True, exist_ok=True)
            
            # Check if already completed
            metrics_file = Path(task.result_dir) / "metrics.json"
            if metrics_file.exists():
                logging.info(f"Actor {self.actor_id}: Skipping completed {task.task_id}")
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return BenchmarkResult(
                    task_id=task.task_id,
                    success=True,
                    metrics=metrics,
                    execution_time=0.0,
                    model_load_time=0.0
                )
            
            # Setup micro metrics
            metric_logger = MicroMetricLogger()
            metric_logger.configure_logging(
                log_path=task.result_dir,
                enabled_metrics=["research_attention_density", "research_attention_output_error"],
                max_records=5000,
                sampling_factor=0.1
            )
            
            # Run benchmark
            logging.info(f"Actor {self.actor_id}: Running {task.task_id}")
            
            metrics = benchmark.run_benchmark(
                adapter=adapter,
                result_dir=task.result_dir,
                generation_kwargs=task.generation_kwargs,
                request_kwargs=task.request_kwargs
            )
            
            metric_logger.flush()
            
            execution_time = time.time() - total_start
            
            return BenchmarkResult(
                task_id=task.task_id,
                success=True,
                metrics=metrics,
                execution_time=execution_time,
                model_load_time=model_load_time
            )
            
        except Exception as e:
            logging.error(f"Actor {self.actor_id}: Task {task.task_id} failed: {e}")
            traceback.print_exc()
            
            return BenchmarkResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - total_start
            )
        
        finally:
            # Always clean up the model to ensure no state leakage
            if adapter is not None:
                logging.info(f"Actor {self.actor_id}: Cleaning up model for {task.task_id}")
                try:
                    del adapter
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.warning(f"Actor {self.actor_id}: Cleanup error: {e}")
    
    def get_stats(self) -> Dict:
        """Return actor statistics."""
        return {
            "actor_id": self.actor_id,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "status": "active"
        }
    
    def cleanup(self):
        """Clean up resources."""
        logging.info(f"Actor {self.actor_id}: Final cleanup")
        torch.cuda.empty_cache()




def load_optimal_configs(config_dir: Path) -> List[BenchmarkTask]:
    """Load optimal configurations and create benchmark tasks."""
    tasks = []
    
    for config_file in config_dir.glob("*.json"):
        if config_file.name.endswith(("_trials.json", "_analysis.csv")):
            continue
        
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
            
            task_id = f"{data['model']}_{data['task']}_{data['masker_name']}".replace("/", "_")
            
            task = BenchmarkTask(
                task_id=task_id,
                model_name=data["model"],
                task_name=data["task"],
                masker_name=data["masker_name"],
                sparse_config=data.get("sparse_config"),
                result_dir="",  # Will be set later
                generation_kwargs={},  # Will be set later
                request_kwargs={}  # Will be set later
            )
            tasks.append(task)
            
        except Exception as e:
            logging.warning(f"Failed to load {config_file}: {e}")
    
    return tasks


@ray.remote
def progress_reporter(total_tasks: int, result_queue: RayQueue) -> None:
    """Ray task that reports progress from result queue."""
    completed = 0
    failed = 0
    start_time = time.time()
    total_model_load_time = 0.0
    
    while completed + failed < total_tasks:
        try:
            result = result_queue.get(timeout=10)
            
            if result.success:
                completed += 1
                total_model_load_time += result.model_load_time
                
                print(f"[{completed + failed}/{total_tasks}] ✓ {result.task_id} "
                      f"({result.execution_time:.1f}s, model load: {result.model_load_time:.1f}s)")
            else:
                failed += 1
                print(f"[{completed + failed}/{total_tasks}] ✗ {result.task_id} - {result.error}")
            
            # Print progress stats every 10 tasks
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + failed) / elapsed
                eta = (total_tasks - completed - failed) / rate if rate > 0 else 0
                avg_load_time = total_model_load_time / max(1, completed)
                print(f"\n--- Progress: {completed + failed}/{total_tasks} "
                      f"({rate:.2f} tasks/s, ETA: {eta/60:.1f} min) ---")
                print(f"--- Avg model load time: {avg_load_time:.1f}s ---\n")
                
        except Exception:
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Total model load time: {total_model_load_time/60:.1f} minutes")
    print(f"Throughput: {completed/total_time*3600:.1f} tasks/hour")


def main(
    configs_dir: str,
    benchmark_results_dir: str = "/data/apdesai/DO_NOT_DELETE/sparse_attention_hub",
    max_new_tokens: int = 1000,
    max_context_length: int = 100000,
    max_requests: int = 100,
    actors_per_gpu: Optional[int] = None
):
    """Ray-based parallel benchmark runner with efficient resource management.
    
    Args:
        configs_dir: Directory containing optimal configurations (e.g., "./optimal_configs/run_20250818_203531")
        benchmark_results_dir: Directory to save benchmark results (default: "./benchmark_vt_full_10pct")
        max_new_tokens: Maximum number of new tokens to generate (default: 1000)
        max_context_length: Maximum context length for requests (default: 100000)
        max_requests: Maximum number of requests per benchmark (default: 1000)
        actors_per_gpu: Number of actors per GPU for better utilization (default: 1)
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print(f"\n{'='*80}")
    print(f"RAY BENCHMARK RUNNER")
    print(f"{'='*80}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Get GPU info
    num_gpus = int(ray.available_resources().get("GPU", 0))
    if num_gpus == 0:
        print("Error: No GPUs available")
        sys.exit(1)
    
    # Determine number of actors
    if actors_per_gpu:
        num_actors = num_gpus * actors_per_gpu
        print(f"Creating {actors_per_gpu} actors per GPU for maximum utilization")
    else:
        # Default to number of GPUs
        num_actors = num_gpus
        
    print(f"Ray cluster: {ray.available_resources()}")
    print(f"Using {num_actors} actors on {num_gpus} GPUs")
    
    # Load configurations
    config_dir = Path(configs_dir)
    if not config_dir.exists():
        print(f"Error: Config directory {config_dir} not found")
        sys.exit(1)
    
    print(f"\nLoading configurations from {config_dir}...")
    tasks = load_optimal_configs(config_dir)
    print(f"Loaded {len(tasks)} configurations")
    
    # Set generation and request parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    
    request_kwargs = {
        "max_context_length": max_context_length,
        "max_requests": max_requests
    }
    
    # Update tasks with full configuration
    for task in tasks:
        task.result_dir = os.path.join(
            benchmark_results_dir,
            task.model_name.replace("/", "_"),
            task.masker_name,
            task.task_name.replace("/", "_")
        )
        task.generation_kwargs = generation_kwargs
        task.request_kwargs = request_kwargs
    
    # Tasks are ready for execution
    print(f"\nReady to execute {len(tasks)} tasks...")
    
    
    # Create adapter config
    adapter_config = {
        "adapter_name": "huggingface",
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "tokenizer_kwargs": {"padding_side": "left"}
    }
    
    # Create Ray actors
    print(f"\nCreating {num_actors} Ray actors...")
    actors = []
    
    # Calculate GPU resources per actor
    if actors_per_gpu and actors_per_gpu > 1:
        # When multiple actors per GPU, each gets a fraction
        gpu_per_actor = 1.0 / actors_per_gpu
        print(f"Each actor will use {gpu_per_actor:.2f} GPU resources")
        
        # Create actors with fractional GPU resources
        for i in range(num_actors):
            # Have to use options to set fractional GPU
            actor = GPUBenchmarkActor.options(num_gpus=gpu_per_actor).remote(i, adapter_config)
            actors.append(actor)
    else:
        # Standard: one actor per GPU
        for i in range(num_actors):
            actor = GPUBenchmarkActor.remote(i, adapter_config)
            actors.append(actor)
    
    # Create result queue and progress reporter
    result_queue = RayQueue(maxsize=len(tasks))
    progress_task = progress_reporter.remote(len(tasks), result_queue)
    
    # Create actor pool for load balancing
    pool = ActorPool(actors)
    
    # Submit all tasks
    print(f"\nSubmitting {len(tasks)} tasks...")
    print("-" * 80)
    
    start_time = time.time()
    
    # Submit tasks to actor pool
    # ActorPool.submit expects (fn, value) where fn(actor, value) is called
    for task in tasks:
        pool.submit(lambda actor, task: actor.run_benchmark.remote(task), task)
    
    # Collect results
    while pool.has_next():
        result = pool.get_next()
        result_queue.put(result)
    
    # Wait for progress reporter
    ray.get(progress_task)
    
    # Get actor statistics
    print("\nActor statistics:")
    for actor in actors:
        stats = ray.get(actor.get_stats.remote())
        print(f"  Actor {stats['actor_id']} ({stats['gpu_name']}): {stats['status']}")
    
    # Cleanup
    print("\nCleaning up...")
    for actor in actors:
        ray.get(actor.cleanup.remote())
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {benchmark_results_dir}")
    print(f"{'='*80}")
    
    ray.shutdown()


if __name__ == "__main__":
    fire.Fire(main)

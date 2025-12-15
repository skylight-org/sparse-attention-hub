"""Profile indexer_first and indexer_next functions for efficient attention backends.

This module profiles the indexer_first or indexer_next methods of EfficientAttention
classes using PyTorch profiler and manual timing measurements.
"""

import argparse
import importlib
import importlib.util
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from ...utils.mask import Mask
from ..base import EfficientAttention, EfficientAttentionConfig
from ..efficient_attention_research_backend import (
    EfficientAttentionResearchBackend,
    EfficientAttentionResearchBackendConfig,
)
from .util import move_sparse_meta_data_to_device


def load_indexer_function_from_file(
    file_path: str, function_name: str
) -> Callable:
    """Dynamically load an indexer function from a Python file.

    Args:
        file_path: Path to the Python file containing the function.
        function_name: Name of the function to load (e.g., '__indexer_first' or '__indexer_next').

    Returns:
        The function from the loaded module.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file does not contain the specified function.
        ImportError: If the file cannot be imported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Indexer file not found: {file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_indexer_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the function
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Module {file_path} does not contain a '{function_name}' function"
        )

    indexer_fn: Callable = getattr(module, function_name)
    return indexer_fn


def replace_indexer_functions_in_module(
    class2: Type[EfficientAttention],
    indexer_first_file: Optional[str] = None,
    indexer_next_file: Optional[str] = None,
) -> None:
    """Replace __indexer_first and/or __indexer_next functions in class2's module.

    Args:
        class2: The class whose module should have its functions replaced.
        indexer_first_file: Optional path to file containing __indexer_first function.
        indexer_next_file: Optional path to file containing __indexer_next function.
    """
    module_name: str = class2.__module__
    module = sys.modules.get(module_name)
    if module is None:
        # Import the module if it's not already loaded
        module = importlib.import_module(module_name)

    if indexer_first_file is not None:
        new_indexer_first: Callable = load_indexer_function_from_file(
            indexer_first_file, "__indexer_first"
        )
        # Update module's globals to replace the function
        module.__dict__["__indexer_first"] = new_indexer_first
        print(f"Replaced __indexer_first in {module_name} with function from {indexer_first_file}")

    if indexer_next_file is not None:
        new_indexer_next: Callable = load_indexer_function_from_file(
            indexer_next_file, "__indexer_next"
        )
        # Update module's globals to replace the function
        module.__dict__["__indexer_next"] = new_indexer_next
        print(f"Replaced __indexer_next in {module_name} with function from {indexer_next_file}")


def profile_indexer_first(
    class1: Type[EfficientAttentionResearchBackend],
    class2: Type[EfficientAttention],
    num_warmup_runs: int = 5,
    num_profile_runs: int = 1,
    num_timing_runs: int = 50,
    indexer_first_file: Optional[str] = None,
) -> Dict[str, float]:
    """Profile the indexer_first function.

    Args:
        class1: Research backend class (EfficientAttentionResearchBackend).
        class2: Native backend class (EfficientAttention).
        num_warmup_runs: Number of warmup runs before profiling.
        num_profile_runs: Number of runs to profile with PyTorch profiler.
        num_timing_runs: Number of runs for manual timing measurements.
        indexer_first_file: Optional path to file containing __indexer_first function to replace.

    Returns:
        Dictionary containing timing statistics.
    """
    B: int = 1
    H: int = 32
    num_keys: int = 32000
    d: int = 128

    # Replace __indexer_first function if file is provided
    if indexer_first_file is not None:
        replace_indexer_functions_in_module(
            class2=class2, indexer_first_file=indexer_first_file
        )

    # Determine device (CUDA if available, else CPU)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get sample data using class1 function create_sample_data_first
    (
        research_attention_config,
        query,
        key,
        value,
        module,
        attention_mask,
        scaling,
        dropout,
        sparse_meta_data,
    ) = class1.create_sample_data_first(B=B, H=H, num_keys=num_keys, d=d)

    # Move tensors and module to device
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    module = module.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    sparse_meta_data = move_sparse_meta_data_to_device(sparse_meta_data, device)

    # Create object2 of class2 using config
    config2_class_name: str = f"{class2.__name__}Config"
    config2_class: Type[EfficientAttentionConfig] = getattr(
        importlib.import_module(class2.__module__), config2_class_name
    )
    config2: EfficientAttentionConfig = config2_class(
        research_attention_config=research_attention_config
    )
    object2: EfficientAttention = class2.create_from_config(config2)

    print(f"\n📊 Profiling indexer_first with parameters:")
    print(f"   B={B}, H={H}, num_keys={num_keys}, d={d}")
    print(f"   Warmup runs: {num_warmup_runs}")
    print(f"   Profile runs: {num_profile_runs}")
    print(f"   Timing runs: {num_timing_runs}")

    # Warmup runs
    print(f"\n🔥 Running {num_warmup_runs} warmup iterations...")
    with torch.no_grad():
        for i in range(num_warmup_runs):
            _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_first(
                query=query,
                key=key,
                value=value,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

    # PyTorch profiler
    print(f"\n📊 Profiling {num_profile_runs} iterations with PyTorch profiler...")
    trace_path: str = "profile_indexer_first_trace.json"
    activities: list = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("indexer_first"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_first(
                        query=query,
                        key=key,
                        value=value,
                        module=module,
                        attention_mask=attention_mask,
                        scaling=scaling,
                        dropout=dropout,
                        sparse_meta_data=sparse_meta_data,
                        layer_idx=0,
                    )

    # Export trace
    prof.export_chrome_trace(trace_path)
    print(f"   Trace saved to: {trace_path}")

    # Print profiling results
    print(f"\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == "cuda":
        print(f"\n📈 Top 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Manual timing for accurate wall-clock measurements
    print(f"\n⏱️ Running manual timing measurements ({num_timing_runs} runs)...")
    times: list = []

    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            # Run indexer_first
            _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_first(
                query=query,
                key=key,
                value=value,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )

            # End timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time: float = time.perf_counter()

            elapsed_time: float = (end_time - start_time) * 1000  # Convert to ms
            times.append(elapsed_time)

    # Calculate statistics
    avg_time: float = statistics.mean(times)
    min_time: float = min(times)
    max_time: float = max(times)
    std_time: float = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time: float = statistics.median(times)

    timing_stats: Dict[str, float] = {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "median_time_ms": median_time,
    }

    print(f"\n⏱️ Timing Statistics:")
    print(f"   - Average time: {timing_stats['avg_time_ms']:.3f} ms")
    print(f"   - Median time:  {timing_stats['median_time_ms']:.3f} ms")
    print(f"   - Min time:     {timing_stats['min_time_ms']:.3f} ms")
    print(f"   - Max time:     {timing_stats['max_time_ms']:.3f} ms")
    print(f"   - Std dev:      {timing_stats['std_time_ms']:.3f} ms")

    print(f"\n🎉 Profiling completed! View trace in https://ui.perfetto.dev/")
    print(f"   Load file: {trace_path}")

    return timing_stats


def profile_indexer_next(
    class1: Type[EfficientAttentionResearchBackend],
    class2: Type[EfficientAttention],
    num_warmup_runs: int = 5,
    num_profile_runs: int = 1,
    num_timing_runs: int = 50,
    indexer_next_file: Optional[str] = None,
) -> Dict[str, float]:
    """Profile the indexer_next function.

    Args:
        class1: Research backend class (EfficientAttentionResearchBackend).
        class2: Native backend class (EfficientAttention).
        num_warmup_runs: Number of warmup runs before profiling.
        num_profile_runs: Number of runs to profile with PyTorch profiler.
        num_timing_runs: Number of runs for manual timing measurements.
        indexer_next_file: Optional path to file containing __indexer_next function to replace.

    Returns:
        Dictionary containing timing statistics.
    """
    B: int = 1
    H: int = 32
    num_keys: int = 256
    d: int = 128

    # Replace __indexer_next function if file is provided
    if indexer_next_file is not None:
        replace_indexer_functions_in_module(
            class2=class2, indexer_next_file=indexer_next_file
        )

    # Determine device (CUDA if available, else CPU)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get sample data using class1 function create_sample_data_next
    (
        research_attention_config,
        query,
        key,
        value,
        module,
        attention_mask,
        scaling,
        dropout,
        sparse_meta_data,
    ) = class1.create_sample_data_next(B=B, H=H, num_keys=num_keys, d=d)

    # Move tensors and module to device
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    module = module.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    sparse_meta_data = move_sparse_meta_data_to_device(sparse_meta_data, device)

    # Create object2 of class2 using config
    config2_class_name: str = f"{class2.__name__}Config"
    config2_class: Type[EfficientAttentionConfig] = getattr(
        importlib.import_module(class2.__module__), config2_class_name
    )
    config2: EfficientAttentionConfig = config2_class(
        research_attention_config=research_attention_config
    )
    object2: EfficientAttention = class2.create_from_config(config2)

    print(f"\n📊 Profiling indexer_next with parameters:")
    print(f"   B={B}, H={H}, num_keys={num_keys}, d={d}")
    print(f"   Warmup runs: {num_warmup_runs}")
    print(f"   Profile runs: {num_profile_runs}")
    print(f"   Timing runs: {num_timing_runs}")

    # Warmup runs
    print(f"\n🔥 Running {num_warmup_runs} warmup iterations...")
    with torch.no_grad():
        for i in range(num_warmup_runs):
            _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_next(
                query=query,
                key=key,
                value=value,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

    # PyTorch profiler
    print(f"\n📊 Profiling {num_profile_runs} iterations with PyTorch profiler...")
    trace_path: str = "profile_indexer_next_trace.json"
    activities: list = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("indexer_next"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_next(
                        query=query,
                        key=key,
                        value=value,
                        module=module,
                        attention_mask=attention_mask,
                        scaling=scaling,
                        dropout=dropout,
                        sparse_meta_data=sparse_meta_data,
                        layer_idx=0,
                    )

    # Export trace
    prof.export_chrome_trace(trace_path)
    print(f"   Trace saved to: {trace_path}")

    # Print profiling results
    print(f"\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device.type == "cuda":
        print(f"\n📈 Top 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Manual timing for accurate wall-clock measurements
    print(f"\n⏱️ Running manual timing measurements ({num_timing_runs} runs)...")
    times: list = []

    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            # Run indexer_next
            _: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_next(
                query=query,
                key=key,
                value=value,
                module=module,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                layer_idx=0,
            )

            # End timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time: float = time.perf_counter()

            elapsed_time: float = (end_time - start_time) * 1000  # Convert to ms
            times.append(elapsed_time)

    # Calculate statistics
    avg_time: float = statistics.mean(times)
    min_time: float = min(times)
    max_time: float = max(times)
    std_time: float = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time: float = statistics.median(times)

    timing_stats: Dict[str, float] = {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "median_time_ms": median_time,
    }

    print(f"\n⏱️ Timing Statistics:")
    print(f"   - Average time: {timing_stats['avg_time_ms']:.3f} ms")
    print(f"   - Median time:  {timing_stats['median_time_ms']:.3f} ms")
    print(f"   - Min time:     {timing_stats['min_time_ms']:.3f} ms")
    print(f"   - Max time:     {timing_stats['max_time_ms']:.3f} ms")
    print(f"   - Std dev:      {timing_stats['std_time_ms']:.3f} ms")

    print(f"\n🎉 Profiling completed! View trace in https://ui.perfetto.dev/")
    print(f"   Load file: {trace_path}")

    return timing_stats


def load_class_from_string(class_path: str) -> Type:
    """Load a class from a string path.

    Args:
        class_path: Full path to the class, e.g., "module.submodule.ClassName".

    Returns:
        The class object.

    Raises:
        ImportError: If the class cannot be imported.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main() -> None:
    """Main function to run profiling.

    Takes class names and function name to profile (indexer_next / indexer_first) and runs profiling.
    """
    parser = argparse.ArgumentParser(
        description="Profile indexer_first or indexer_next methods of efficient attention backends."
    )
    parser.add_argument(
        "--class1",
        type=str,
        required=True,
        help="Full path to research backend class, e.g., "
        "'sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.research.StreamingLLMResearchBackend'",
    )
    parser.add_argument(
        "--class2",
        type=str,
        required=True,
        help="Full path to native backend class, e.g., "
        "'sparse_attention_hub.sparse_attention.efficient_attention.implementations.streamingllm.native.StreamingLLMNativeBackend'",
    )
    parser.add_argument(
        "--function",
        type=str,
        required=True,
        choices=["indexer-first", "indexer-next"],
        help="Function name to profile: 'indexer-first' or 'indexer-next'",
    )
    parser.add_argument(
        "--indexer-first-file",
        type=str,
        default=None,
        help="Path to Python file containing __indexer_first function to replace in class2. "
        "If specified, the __indexer_first function in class2's module will be replaced.",
    )
    parser.add_argument(
        "--indexer-next-file",
        type=str,
        default=None,
        help="Path to Python file containing __indexer_next function to replace in class2. "
        "If specified, the __indexer_next function in class2's module will be replaced.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs before profiling (default: 5)",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Number of runs to profile with PyTorch profiler (default: 1)",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=50,
        help="Number of runs for manual timing measurements (default: 50)",
    )

    args = parser.parse_args()

    # Load classes
    class1: Type[EfficientAttentionResearchBackend] = load_class_from_string(args.class1)
    class2: Type[EfficientAttention] = load_class_from_string(args.class2)

    # Run the appropriate profiling function
    if args.function == "indexer-first":
        timing_stats: Dict[str, float] = profile_indexer_first(
            class1=class1,
            class2=class2,
            num_warmup_runs=args.warmup_runs,
            num_profile_runs=args.profile_runs,
            num_timing_runs=args.timing_runs,
            indexer_first_file=args.indexer_first_file,
        )
    else:  # indexer-next
        timing_stats = profile_indexer_next(
            class1=class1,
            class2=class2,
            num_warmup_runs=args.warmup_runs,
            num_profile_runs=args.profile_runs,
            num_timing_runs=args.timing_runs,
            indexer_next_file=args.indexer_next_file,
        )

    print(f"\n✅ Profiling of {args.function} completed successfully!")
    print(f"   Average time: {timing_stats['avg_time_ms']:.3f} ms")


if __name__ == "__main__":
    main()

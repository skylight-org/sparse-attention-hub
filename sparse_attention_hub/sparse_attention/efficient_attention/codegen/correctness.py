"""Correctness checks for indexer methods between research and native backends.

This module compares the outputs of indexer_first and indexer_next methods
between EfficientAttentionResearchBackend and EfficientAttention (native backend)
implementations.
"""

import argparse
import importlib
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import nn

from ...utils.mask import Mask
from ..base import EfficientAttention, EfficientAttentionConfig
from ..efficient_attention_research_backend import (
    EfficientAttentionResearchBackend,
    EfficientAttentionResearchBackendConfig,
)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str = "tensor",
) -> bool:
    """Compare two tensors with appropriate tolerance based on dtype.

    Args:
        tensor1: First tensor to compare.
        tensor2: Second tensor to compare.
        name: Name of the tensor for error messages.

    Returns:
        True if tensors match, False otherwise.

    Raises:
        AssertionError: If tensors don't match.
    """
    if tensor1.shape != tensor2.shape:
        raise AssertionError(
            f"{name} shape mismatch: {tensor1.shape} vs {tensor2.shape}"
        )

    # Determine tolerance based on dtype: exact match for int, 0.01 for float
    if tensor1.dtype.is_floating_point or tensor2.dtype.is_floating_point:
        tolerance: float = 0.01
        if not torch.allclose(tensor1, tensor2, atol=tolerance, rtol=tolerance):
            max_diff: float = float(torch.max(torch.abs(tensor1 - tensor2)).item())
            raise AssertionError(
                f"{name} values don't match (max_diff={max_diff}, tolerance={tolerance})"
            )
    else:
        # Integer types: exact match
        if not torch.equal(tensor1, tensor2):
            raise AssertionError(f"{name} integer values don't match exactly")

    return True


def check_indexer_first_correctness(
    class1: Type[EfficientAttentionResearchBackend],
    class2: Type[EfficientAttention],
    num_iterations: int = 10,
) -> bool:
    """Test that indexer_first from both classes match.

    Args:
        class1: Research backend class (EfficientAttentionResearchBackend).
        class2: Native backend class (EfficientAttention, typically EfficientAttentionNativeBackend).
        num_iterations: Number of test iterations (default: 10).

    Returns:
        True if all tests pass, False otherwise.
    """
    B: int = 2
    H: int = 4
    num_keys: int = 64
    d: int = 32

    # Determine device (CUDA if available, else CPU)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_iterations):
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

        # Create object1 of class1 using given research_attention_config
        # Get the config class from the same module as class1
        config1_class_name: str = f"{class1.__name__}Config"
        config1_class: Type[EfficientAttentionResearchBackendConfig] = getattr(
            importlib.import_module(class1.__module__), config1_class_name
        )
        config1: EfficientAttentionResearchBackendConfig = config1_class(
            research_attention_config=research_attention_config
        )
        object1: EfficientAttentionResearchBackend = class1.create_from_config(config1)

        # Create object2 of class2 using config
        # For native backend, we need to create a config with the same research_attention_config
        config2_class_name: str = f"{class2.__name__}Config"
        config2_class: Type[EfficientAttentionConfig] = getattr(
            importlib.import_module(class2.__module__), config2_class_name
        )
        config2: EfficientAttentionConfig = config2_class(
            research_attention_config=research_attention_config
        )
        object2: EfficientAttention = class2.create_from_config(config2)

        # results1 = object1.indexer_first on the data (research backend returns Mask)
        results1: Mask = object1.indexer_first(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
        )

        # results2 = object2.indexer_first on the data (native backend returns tuple)
        results2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_first(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
        )

        # Use check_correctness_with_research_backend to compare
        # This method handles the format conversion and comparison internally
        is_correct: bool = object2.check_correctness_with_research_backend(results1, *results2)
        if not is_correct:
            raise AssertionError(
                f"Correctness check failed for indexer_first (iteration {i})"
            )

    return True


def check_indexer_next_correctness(
    class1: Type[EfficientAttentionResearchBackend],
    class2: Type[EfficientAttention],
    num_iterations: int = 10,
) -> bool:
    """Test that indexer_next from both classes match.

    Args:
        class1: Research backend class (EfficientAttentionResearchBackend).
        class2: Native backend class (EfficientAttention, typically EfficientAttentionNativeBackend).
        num_iterations: Number of test iterations (default: 10).

    Returns:
        True if all tests pass, False otherwise.
    """
    B: int = 2
    H: int = 4
    num_keys: int = 64
    d: int = 32

    # Determine device (CUDA if available, else CPU)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_iterations):
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

        # Create object1 of class1 using given research_attention_config
        # Get the config class from the same module as class1
        config1_class_name: str = f"{class1.__name__}Config"
        config1_class: Type[EfficientAttentionResearchBackendConfig] = getattr(
            importlib.import_module(class1.__module__), config1_class_name
        )
        config1: EfficientAttentionResearchBackendConfig = config1_class(
            research_attention_config=research_attention_config
        )
        object1: EfficientAttentionResearchBackend = class1.create_from_config(config1)

        # Create object2 of class2 using config
        # For native backend, we need to create a config with the same research_attention_config
        config2_class_name: str = f"{class2.__name__}Config"
        config2_class: Type[EfficientAttentionConfig] = getattr(
            importlib.import_module(class2.__module__), config2_class_name
        )
        config2: EfficientAttentionConfig = config2_class(
            research_attention_config=research_attention_config
        )
        object2: EfficientAttention = class2.create_from_config(config2)

        # results1 = object1.indexer_next on the data (research backend returns Mask)
        results1: Mask = object1.indexer_next(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
        )

        # results2 = object2.indexer_next on the data (native backend returns tuple)
        results2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = object2.indexer_next(
            query=query,
            key=key,
            value=value,
            module=module,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
        )

        # Use check_correctness_with_research_backend to compare
        # This method handles the format conversion and comparison internally
        is_correct: bool = object2.check_correctness_with_research_backend(results1, *results2)
        if not is_correct:
            raise AssertionError(
                f"Correctness check failed for indexer_next (iteration {i})"
            )

    return True


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
    """Main function to run correctness tests.

    Takes class names and function name to test (indexer_next / indexer_first) and runs the test.
    """
    parser = argparse.ArgumentParser(
        description="Test correctness of indexer methods between research and native backends."
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
        choices=["indexer_first", "indexer_next"],
        help="Function name to test: 'indexer_first' or 'indexer_next'",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)",
    )

    args = parser.parse_args()

    # Load classes
    class1: Type[EfficientAttentionResearchBackend] = load_class_from_string(args.class1)
    class2: Type[EfficientAttention] = load_class_from_string(args.class2)

    # Run the appropriate test
    if args.function == "indexer_first":
        result: bool = check_indexer_first_correctness(
            class1=class1, class2=class2, num_iterations=args.num_iterations
        )
    else:  # indexer_next
        result: bool = check_indexer_next_correctness(
            class1=class1, class2=class2, num_iterations=args.num_iterations
        )

    if result:
        print(f"✅ All {args.num_iterations} iterations of {args.function} correctness test passed!")
    else:
        print(f"❌ {args.function} correctness test failed!")
        exit(1)


if __name__ == "__main__":
    main()


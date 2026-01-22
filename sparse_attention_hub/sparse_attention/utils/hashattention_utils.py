import pickle
from typing import Dict, List, cast

import torch

""" This only works for 3 layered MLPs. Need to fix for other cases"""


def convert_usa_weights_to_hash_attention(
    usa_checkpoint_path: str,
    num_layers: int = 32,
    num_heads: int = 32,
    num_mlp_layers: int = 3,
    device: str = "cpu",
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Convert USA module weights to HashAttentionTopKMasker format.

    Args:
        usa_checkpoint_path: Path to USA checkpoint file
        num_layers: Number of layers in the model
        num_heads: Number of attention heads
        num_mlp_layers: Number of MLP layers (default: 3)
        device: Device to load weights on

    Returns:
        Dictionary of layer-wise weights in HashAttention format
    """

    print(f"Loading USA weights from {usa_checkpoint_path}")
    usa_state_dict = torch.load(usa_checkpoint_path, map_location=device)

    hat_weights = {}

    for layer_idx in range(num_layers):
        layer_weights: Dict[str, List[torch.Tensor]] = {
            "query_matrix": [],
            "query_bias": [],
            "key_matrix": [],
            "key_bias": [],
        }

        # Collect weights for all heads in this layer
        query_matrices_per_layer: List[List[torch.Tensor]] = [
            [] for _ in range(num_mlp_layers)
        ]
        query_biases_per_layer: List[List[torch.Tensor]] = [
            [] for _ in range(num_mlp_layers)
        ]
        key_matrices_per_layer: List[List[torch.Tensor]] = [
            [] for _ in range(num_mlp_layers)
        ]
        key_biases_per_layer: List[List[torch.Tensor]] = [
            [] for _ in range(num_mlp_layers)
        ]

        for head_idx in range(num_heads):
            query_prefix = f"{layer_idx}.learning_to_hash_transformation_q.{head_idx}"
            key_prefix = f"{layer_idx}.learning_to_hash_transformation_k.{head_idx}"

            # Extract weights from MLP layers (linear layers at indices 0, 2, 4, ...)
            linear_indices = [i * 2 for i in range(num_mlp_layers)]
            for i, linear_idx in enumerate(linear_indices):
                # Query weights
                weight_key = f"{query_prefix}.{linear_idx}.weight"
                bias_key = f"{query_prefix}.{linear_idx}.bias"

                if weight_key in usa_state_dict:
                    weight = usa_state_dict[
                        weight_key
                    ].t()  # Transpose to (in_features, out_features)
                    query_matrices_per_layer[i].append(weight)

                    if bias_key in usa_state_dict:
                        query_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        query_biases_per_layer[i].append(
                            torch.zeros(usa_state_dict[weight_key].shape[0])
                        )

                # Key weights
                weight_key = f"{key_prefix}.{linear_idx}.weight"
                bias_key = f"{key_prefix}.{linear_idx}.bias"

                if weight_key in usa_state_dict:
                    weight = usa_state_dict[
                        weight_key
                    ].t()  # Transpose to (in_features, out_features)
                    key_matrices_per_layer[i].append(weight)

                    if bias_key in usa_state_dict:
                        key_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        key_biases_per_layer[i].append(
                            torch.zeros(usa_state_dict[weight_key].shape[0])
                        )

        # Stack all heads for each layer
        for i in range(num_mlp_layers):
            if query_matrices_per_layer[i]:
                layer_weights["query_matrix"].append(
                    torch.stack(query_matrices_per_layer[i])
                )
                layer_weights["query_bias"].append(
                    torch.stack(query_biases_per_layer[i])
                )
                layer_weights["key_matrix"].append(
                    torch.stack(key_matrices_per_layer[i])
                )
                layer_weights["key_bias"].append(torch.stack(key_biases_per_layer[i]))

        hat_weights[layer_idx] = layer_weights

    print(
        f"✅ Converted weights for {num_layers} layers, {num_heads} heads, {num_mlp_layers} MLP layers"
    )
    return hat_weights


def create_hat_weights_file_from_usa(
    usa_checkpoint_path: str,
    target_hat_path: str,
    num_layers: int = 32,
    num_heads: int = 32,
    num_mlp_layers: int = 3,
    device: str = "cpu",
) -> None:
    """Create HAT weights file from USA checkpoint.

    Args:
        usa_checkpoint_path: Path to USA checkpoint file
        target_hat_path: Path where HAT weights file will be saved
        num_layers: Number of layers in the model
        num_heads: Number of attention heads
        num_mlp_layers: Number of MLP layers
        device: Device to load weights on
    """
    print("Creating HAT weights file from USA checkpoint...")

    # Convert USA weights to HAT format
    hat_weights = convert_usa_weights_to_hash_attention(
        usa_checkpoint_path=usa_checkpoint_path,
        num_layers=num_layers,
        num_heads=num_heads,
        num_mlp_layers=num_mlp_layers,
        device=device,
    )

    # Save to pickle file
    with open(target_hat_path, "wb") as f:
        pickle.dump(hat_weights, f)

    print(f"✅ HAT weights saved to {target_hat_path}")


def load_hat_weights(
    hat_weights_path: str, device: str = "cpu"
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Load HAT weights from pickle file.

    Args:
        hat_weights_path: Path to HAT weights pickle file
        device: Device to load weights on

    Returns:
        Dictionary of layer-wise weights in HashAttention format
    """
    print(f"Loading HAT weights from {hat_weights_path}")

    with open(hat_weights_path, "rb") as f:
        hat_weights: Dict[int, Dict[str, List[torch.Tensor]]] = cast(
            Dict[int, Dict[str, List[torch.Tensor]]], pickle.load(f)
        )

    # Move weights to specified device
    for layer_idx, layer_weights in hat_weights.items():
        for key, value in layer_weights.items():
            hat_weights[layer_idx][key] = [tensor.to(device) for tensor in value]

    print(f"✅ Loaded HAT weights for {len(hat_weights)} layers")
    return hat_weights

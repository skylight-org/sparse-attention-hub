"""Run configuration for hyperparameter search.

All configuration parameters for the hyperparameter search are defined here.
Modify this file to change search behavior without editing the main script.
"""

import os
from typing import Dict, List, Optional

# Model configurations
# Weight files are loaded from SPARSE_ATTENTION_WEIGHTS_DIR environment variable
# Set it to the directory containing your HashAttention weight files

HASHATTENTION_WEIGHTS_DIR: str = "/data/apdesai/code/HashAttention-1.0/artifacts"
DOUBLE_SPARSITY_CONFIG_DIR: str = "/data/apdesai/code/DoubleSparse/config"
hashattention_dir: str = HASHATTENTION_WEIGHTS_DIR
doublesparsity_config_dir: str = DOUBLE_SPARSITY_CONFIG_DIR


MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "llama3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "llama3.1-8b-patch.64K.v1.hat_weights.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.1-8B-Instruct.json"),
    },
    "llama3.2-1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.2-1B-Instruct.json"),
    },
    "llama3.2-3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "meta-llama/Llama-3.2-3B-Instruct.json"),
    },
    "deepseek": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K_hat_weights.pkl"),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "Mistral-7B-Instruct-v0.3.24K.20.500.hat_weights.pkl"),
    },
    "qwen3-30b-moe": {
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "Qwen/Qwen3-30B-A3B-Instruct-2507.json"),
    },
    "qwen3-4b": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "hash_attention_weight_file": os.path.join(hashattention_dir, "DNE.pkl"),
        "double_sparsity_config_file": os.path.join(doublesparsity_config_dir, "Qwen/Qwen3-4B-Instruct-2507.json"), 
    },
}

MODELS : List[str] = [
    "llama3.1-8b", 
    "llama3.2-1b",
    "llama3.2-3b",
    "qwen3-4b",
    "qwen3-30b-moe",
]

TASKS: List[str] = [
    # "ruler32k/vt",
    # "ruler32k/qa_1",
    # "ruler32k/qa_2",
    # "ruler32k/fwe",
    # "ruler32k/niah_multikey_2",
    "ruler32k/niah_multikey_3",
]

SPARSITY_OBJECTIVES: List[str] = [
    2,
    5,
    10,
    20,
]

MEMORY_OBJECTIVES: List[Optional[str]] = [
    32,
    64,
    128,
] # Memory objective parameter (e.g., "memory_32") for configs that need it

BUILDER_NAMES: List[str] = [
    # "dense",
    # "double_sparsity", 
    # "hashattention_topk",
    "magicpig",
    # "oracle_topk",
    # "oracle_topp", 
    # "quest_topk",
    # "vattention_hashattention",
    # "vattention_oracle",
]  # Specify which builders to use (e.g., ["magicpig"], ["dense"], ["double_sparsity"])


# SEARCH PARAMS
NUM_SAMPLES: int = 1  # Number of samples per hyperparameter search
SEARCH_MAX_NEW_TOKENS: int = 3  # Max new tokens for search trials
SEARCH_MAX_CONTEXT_LENGTH: int = 40000  # Max context length for search trials
SEARCH_MAX_REQUESTS: int = 3  # Max requests per search trial
OPTIMAL_CONFIGS_DIR: str = "/data/apdesai/DO_NOT_DELETE/magicpig_optimization"  # Directory for storing optimal configurations
RAY_RESULTS_DIR: str = "/tmp/ray_results"  # Directory for Ray Tune results
SEARCH_TIMEOUT: int = 900  # Timeout per search trial in seconds
ACTORS_PER_GPU: int = 1  # Number of actors per GPU for resource allocation


""" DRY RUN 
if true , it will do everything except the actual running of benchmark helper -- it will just return 
randomly generated scores for each trial and choose based on that
"""
DRY_RUN: bool = False 


""" If you use Time stamp then by default it will perform entire search again.
"""
USE_TIMESTAMP_FOR_RESULTS_DIR: bool = False
FORCE_SEARCH: bool = False # Force re-run of search even if configs exist


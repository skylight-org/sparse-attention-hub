#!/usr/bin/env python3
"""
Interactive Chat Script for Sparse Attention Models

Usage:
    python3 sparse_attention_hub/scripts/chat.py --sparse_attention_config config.yaml --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys
import os
import torch
import yaml
from pathlib import Path

# Ensure we're in the correct directory and add to Python path
# This dynamically finds the repo root relative to this script
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    PQCacheConfig
)
from sparse_attention_hub.adapters import ModelAdapterHF

def get_multiline_input(label):
    print(f"$ {label} =")
    print("(Enter text. Type 'EOF' on a new line to finish)")
    lines = []
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)

def load_config(config_path):
    """
    Loads the sparse attention configuration from a YAML file for the PQCache masker.
    Expected YAML structure (example):
    masker_configs:
      - max_keys: 65536
        key_dim: 64
        pq_clusters: 256
        sorted_channel_file: "..."
        channel_selection: "q_proj"
    The keys must match the constructor kwargs of PQCacheConfig.
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    masker_configs = []
    raw_maskers = raw_config.get('masker_configs', []) if isinstance(raw_config, dict) else raw_config
    
    for m_cfg in raw_maskers:
        if not isinstance(m_cfg, dict):
            raise ValueError("Each masker entry must be a mapping of PQCache args.")
        # Build PQCache config only
        masker_configs.append(PQCacheConfig(**m_cfg))
        
    return ResearchAttentionConfig(masker_configs=masker_configs)

def main():
    parser = argparse.ArgumentParser(description="Chat with Sparse Attention Model")
    parser.add_argument("--sparse_attention_config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", required=True, help="Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    #first we wnat to load the config from user
    print(f"Loading config from {args.sparse_attention_config}...")
    sparse_attention_config = load_config(args.sparse_attention_config)

    #device and attn Implementation
    if torch.cuda.is_available():
        device = "cuda"
        # Use flash_attention_3 if supported (Hopper), else 2
        attn_impl = "flash_attention_2" 
    elif torch.backends.mps.is_available():
        device = "mps"
        attn_impl = "eager" #mps often requires eager or sdpa
    else:
        device = "cpu"
        attn_impl = "eager"

    print(f"Loading model {args.model} on {device}...")

    #init Adapter
    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32, 
            "attn_implementation": attn_impl
        },
        device=device
    )

    #interactive input
    context = get_multiline_input("context")
    question = get_multiline_input("question")

    #prepare the  prompt for user
    #use chat template if available for best results with Instruct models
    if hasattr(adapter.tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
        prompt = adapter.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.model.device)

    # 6. Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = adapter.model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    response = adapter.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\nResponse:")

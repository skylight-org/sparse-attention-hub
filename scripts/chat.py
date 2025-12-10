#!/usr/bin/env python3
"""
Interactive Chat Script for Sparse Attention Models

Usage:
    python3 scripts/chat.py --sparse_attention_config config.yaml --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys
import os
import torch
import yaml
from pathlib import Path
from transformers import BitsAndBytesConfig # Added for better VRAM management

# --- Dynamic Path Insertion ---
# Ensure we're in the correct directory and add to Python path
# This dynamically finds the repo root relative to this script
current_file = Path(__file__).resolve()
# Assumes the script is in /root/scripts/chat.py and the root is /root/
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Check for required imports from your project structure
try:
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        PQCacheConfig
    )
    from sparse_attention_hub.adapters import ModelAdapterHF
except ImportError as e:
    print("="*60, file=sys.stderr)
    print(f"FATAL IMPORT ERROR: Could not find required modules.", file=sys.stderr)
    print(f"Please check your Python path and project structure.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    sys.exit(1)


def get_multiline_input(label):
    """Gets multiline input from the user until 'EOF' is typed."""
    print(f"\n$ {label} =")
    print("(Enter text. Type 'EOF' on a new line to finish)")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "EOF":
                break
            lines.append(line)
        except EOFError:
            # Handle Ctrl+D
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nInput cancelled.", file=sys.stderr)
            return None
    return "\n".join(lines)

def load_config(config_path):
    """
    Loads the sparse attention configuration from a YAML file.
    """
    # Use Path to handle relative/absolute paths robustly
    resolved_path = Path(config_path).resolve() 
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")
        
    with open(resolved_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    masker_configs = []
    # Assumes the PQCacheConfig kwargs are provided under 'masker_configs'
    raw_maskers = raw_config.get('masker_configs', []) if isinstance(raw_config, dict) else raw_config
    
    for m_cfg in raw_maskers:
        if not isinstance(m_cfg, dict):
            raise ValueError("Each masker entry must be a mapping of PQCache args.")
        # Attempt to build PQCache config
        masker_configs.append(PQCacheConfig(**m_cfg))
            
    return ResearchAttentionConfig(masker_configs=masker_configs)

def main():
    parser = argparse.ArgumentParser(description="Chat with Sparse Attention Model")
    parser.add_argument("--sparse_attention_config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", required=True, help="Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"Loading config from {args.sparse_attention_config}...")
    try:
        sparse_attention_config = load_config(args.sparse_attention_config)
    except Exception as e:
        print(f"\n[FATAL CONFIG ERROR] Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Device and Attention Implementation Setup
    if torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 for better numerical stability and speed on modern GPUs
        dtype = torch.bfloat16
        # Use Flash Attention if supported, which is crucial for speed
        attn_impl = "flash_attention_2"
        # 4-bit quantization config (crucial for VRAM management)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
        print(f"Using GPU ({device}) with bfloat16 and 4-bit quantization.")
    else:
        device = "cpu"
        dtype = torch.float32
        attn_impl = "eager"
        quant_config = None
        print(f"WARNING: Running on CPU ({device}). Performance will be very slow.")


    # 3. Initialize Adapter and Load Model (Crucial Error Handling Block)
    print(f"Loading model {args.model}...")
    try:
        # NOTE: ModelAdapterHF is assumed to handle the patching of attention layers internally
        adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_attention_config,
            model_kwargs={
                "torch_dtype": dtype, 
                "attn_implementation": attn_impl,
                "device_map": "auto", # Auto device mapping for efficient VRAM usage
                "quantization_config": quant_config, # Pass 4-bit config if available
                # If running Llama 3/4 from HF, authentication is required
                "token": os.environ.get("HUGGINGFACE_TOKEN") # Pass token if set in environment
            },
            device=device
        )
        print("Model Adapter initialized successfully.")
    
    except Exception as e:
        # This block catches errors during model loading, patching, or OOM
        print("\n" + "="*70, file=sys.stderr)
        print(f"FATAL MODEL/ADAPTER INITIALIZATION ERROR:", file=sys.stderr)
        print(f"Type: {type(e).__name__}", file=sys.stderr)
        print(f"Message: {e}", file=sys.stderr)
        print("\nCheck 1: VRAM/Quantization (Did you see 'CUDA out of memory'?)", file=sys.stderr)
        print("Check 2: Model Access (Did you run 'huggingface-cli login'?)", file=sys.stderr)
        print("Check 3: Patching (Is the PQCache patch compatible with Llama 3.1?)", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        raise # Re-raise to provide full traceback for debugging

    # --- 4. Interactive Input Loop ---
    print("\n--- Sparse Attention Chat Initiated ---")
    
    while True:
        try:
            # Get user input
            context = get_multiline_input("context")
            if context is None: # Input cancelled (Ctrl+C)
                break 
            
            question = get_multiline_input("question")
            if question is None:
                break
            
            if not context.strip() and not question.strip():
                 print("Input empty, starting new prompt.", file=sys.stderr)
                 continue

            # Prepare the prompt
            if hasattr(adapter.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
                ]
                prompt = adapter.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

            inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.model.device)

            # 5. Generate
            print("\nGenerating response...")
            with torch.no_grad():
                outputs = adapter.model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    # Crucial: The custom attention_mask is handled internally by adapter.model 
                    # based on the sparse_attention_config passed during init.
                )

            # 6. Decode and Print Response
            response = adapter.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print("\n$ <Response>:")
            print(response.strip())

        except KeyboardInterrupt:
            # Allow clean exit during input or generation
            print("\nChat session interrupted by user.")
            break
        except Exception as e:
            print(f"\n[RUNTIME ERROR] An error occurred during generation: {e}", file=sys.stderr)
            # Continue loop on runtime error
            continue

    print("\nChat session ended.")

if __name__ == "__main__":
    main()

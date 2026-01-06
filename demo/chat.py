#!/usr/bin/env python3
"""
Interactive Chat Script for Sparse Attention Models

This script loads a model and a sparse attention configuration defined below,
then starts an interactive chat session.

Usage:
    python3 scripts/chat.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys
import os
import torch
from pathlib import Path
#from transformers import BitsAndBytesConfig # Added for better VRAM management

# ==============================================================================
# USER CONFIGURATION SECTION
# 
# Define the specific Sparse Attention setup you want to use here.
# This structure is what is used by the sparse_attention_hub.
# ==============================================================================

#You must ensure all masker config classes used here are imported below.
try:
    #assume these are available in the environment path
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        PQCacheConfig,
        SinkMaskerConfig, 
        LocalMaskerConfig, 
    )
    from sparse_attention_hub.adapters import ModelAdapterHF
except ImportError as e:
    print("="*60, file=sys.stderr)
    print(f"FATAL IMPORT ERROR: Could not find required sparse attention modules.", file=sys.stderr)
    print(f"Please check your Python path and project structure.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    sys.exit(1)


sparse_attention_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(
            sink_size=128,
        ),
        LocalMaskerConfig(
            window_size=128,
        ),
        PQCacheConfig(
            heavy_size=0.1,
            pq_group_factor=2,
            pq_bits=6,
            kmeans_iter=10,
            init_offset=128,
            metric="euclidean",
        ),
    ]
)

# ==============================================================================
# END USER CONFIGURATION SECTION
##If you run into memory issues, consider using the following (and import it):
#quant_config = BitsAndBytesConfig(
#            load_in_4bit=True,
#            bnb_4bit_use_double_quant=True,
#            bnb_4bit_quant_type="nf4",
#            bnb_4bit_compute_dtype=dtype
#        )
# ==============================================================================


# --- Dynamic Path Insertion ---
current_file = Path(__file__).resolve()
#assume the script is in /root/scripts/chat.py and the root is /root/
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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


def main():
    parser = argparse.ArgumentParser(description="Chat with Sparse Attention Model")
    parser.add_argument("--model", required=True, help="Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    print(f"Using sparse attention configuration: {sparse_attention_config.__class__.__name__}")
    if hasattr(sparse_attention_config, 'masker_configs'):
        print("Maskers used:")
        for cfg in sparse_attention_config.masker_configs:
            print(f"- {cfg.__class__.__name__}")


    # device and Attention Implementation Setup
    if torch.cuda.is_available():
        device = "cuda"
        #use bfloat16 for better numerical stability and speed on modern GPUs
        dtype = torch.bfloat16
        #usse Flash Attention if supported, which is crucial for speed. rmr to add this to requirements
        attn_impl = "flash_attention_2"
        quant_config = None
        print(f"Using GPU ({device}) with bfloat16 (Full Precision, NO Quantization).")
    else:
        device = "cpu"
        dtype = torch.float32
        attn_impl = "eager"
        quant_config = None
        print(f"WARNING: Running on CPU ({device}). Performance will be very slow.")


    #initialize Adapter and Load Model (Crucial Error Handling Block)
    print(f"Loading model {args.model}...")
    try:
        #ModelAdapterHF is assumed to handle the patching of attention layers internally
        adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_attention_config,
            model_kwargs={
                "torch_dtype": dtype, 
                "attn_implementation": attn_impl,
                "device_map": "auto", # Auto device mapping for efficient VRAM usage
                "quantization_config": quant_config, # Pass 4-bit config if available
                #if running Llama 3/4 from HF, authentication is required
                "token": os.environ.get("HUGGINGFACE_TOKEN") # Pass token if set in environment
            },
            device=device
        )
        print("Model Adapter initialized successfully.")
    
    except Exception as e:
        #catches errors during model loading, patching, or OOM
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

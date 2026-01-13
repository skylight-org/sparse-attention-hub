#!/usr/bin/env python3
""" OpenAI-compatible API server for sparse-attention-hub models."""

import argparse
import os
import sys
import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
import torch
from flask import Flask, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, PQCacheConfig, SinkMaskerConfig
)

app = Flask(__name__)
adapter = None

# Support Dense and PQCache configurations
CONFIGS = {
    "dense": None,
    "pqcache": ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        PQCacheConfig(heavy_size=0.1, pq_group_factor=2, pq_bits=6, kmeans_iter=10, init_offset=128, metric="euclidean")
    ]),
}

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from model output (JSON or XML formats)."""
    tool_calls = []
    # Standard Qwen <tool_call> tags
    for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        content = match.group(1).strip()
        try:
            if content.startswith('{'): # JSON format
                d = json.loads(content)
                tool_calls.append({"id": f"call_{os.urandom(2).hex()}", "type": "function", 
                                 "function": {"name": d["name"], "arguments": json.dumps(d["arguments"])}})
            else: # XML format
                fn_match = re.search(r'<function=(.*?)>', content)
                if fn_match:
                    args = {p.group(1): p.group(2).strip() for p in re.finditer(r'<parameter=(.*?)>(.*?)</parameter>', content, re.DOTALL)}
                    tool_calls.append({"id": f"call_{os.urandom(2).hex()}", "type": "function", 
                                     "function": {"name": fn_match.group(1), "arguments": json.dumps(args)}})
        except Exception: continue
    return tool_calls

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    global adapter
    data = request.json or {}
    messages = data.get("messages", [])
    tools = data.get("tools")
    
    # Process tools and messages for the template
    if not tools or not isinstance(tools, list): tools = None
    formatted_messages = [{"role": m.get("role"), "content": m.get("content")} for m in messages]
    # Flatten multimodal content to text if needed
    for m in formatted_messages:
        if isinstance(m["content"], list):
            m["content"] = "".join(b.get("text", "") if isinstance(b, dict) else b for b in m["content"])

    # Prepare prompt
    try:
        prompt = adapter.tokenizer.apply_chat_template(formatted_messages, tools=tools, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = adapter.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
    
    inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.device)
    print(f"--- Request: {len(messages)} msgs, {inputs.input_ids.shape[1]} tokens ---")

    # Generate
    with torch.no_grad():
        context_mgr = adapter.enable_sparse_mode() if adapter._sparse_attention_available else torch.no_grad()
        with context_mgr:
            outputs = adapter.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=data.get("max_tokens", 1024),
                do_sample=data.get("temperature", 0.0) > 0,
                temperature=data.get("temperature", 0.0),
                top_p=data.get("top_p", 0.9),
                pad_token_id=adapter.tokenizer.pad_token_id,
                eos_token_id=adapter.tokenizer.eos_token_id,
                **( {"sparse_meta_data": {}} if adapter._sparse_attention_available else {} )
            )
    
    raw_text = adapter.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    tool_calls = extract_tool_calls(raw_text) if tools else []
    
    # Clean text for response
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', raw_text, flags=re.DOTALL).strip()
    clean_text = clean_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    return jsonify({
        "id": "chatcmpl-sparse",
        "object": "chat.completion",
        "model": data.get("model", "sparse-model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": clean_text or None, "tool_calls": tool_calls or None},
            "finish_reason": "tool_calls" if tool_calls else "stop"
        }]
    })

def main():
    global adapter
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("port", type=int, default=4000)
    parser.add_argument("--config", type=str, default="dense", choices=list(CONFIGS.keys()))
    args = parser.parse_args()

    print(f"🚀 Loading {args.model} with config '{args.config}'...")
    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=CONFIGS[args.config],
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda:0", "attn_implementation": "sdpa"},
    )

    # Device synchronization fix for multi-GPU setups
    model_dev = next(adapter.model.parameters()).device
    adapter.device = model_dev
    _orig_forward = adapter.model.forward
    def _wrapped_forward(*args, **kwargs):
        new_args = tuple(a.to(model_dev) if isinstance(a, torch.Tensor) else a for a in args)
        new_kwargs = {k: v.to(model_dev) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return _orig_forward(*new_args, **new_kwargs)
    adapter.model.forward = _wrapped_forward

    print(f"📡 Server ready at http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OpenAI-compatible server for sparse attention models.

Usage:
    python scripts/start_server.py model_name [and] port
"""

import argparse
import json
import os
import re
import sys
import types
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to sys.path to allow importing from sparse_attention_hub
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from sparse_attention_hub.adapters import ModelAdapterHF
    from sparse_attention_hub.sparse_attention.research_attention import (
        ResearchAttentionConfig,
    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        LocalMaskerConfig,
        OracleTopKConfig,
        SinkMaskerConfig,
    )
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print(
        "Ensure you are running from the project root and have all dependencies installed."
    )
    sys.exit(1)

# ==============================================================================
# CONFIGURATION AREA
# Modify this section to change the model's attention behavior.
# ==============================================================================

# Option 1: Standard Dense Attention
# SPARSE_CONFIG = None

# Option 2: Sparse Attention (e.g., Oracle Top-K)
SPARSE_CONFIG: Optional[ResearchAttentionConfig] = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=0.5),
    ]
)

# ==============================================================================

app = FastAPI(title="Sparse Attention Model Server")
model_adapter: Optional[ModelAdapterHF] = None

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    repetition_penalty: Optional[float] = 1.05
    max_tokens: Optional[int] = 65536
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = "allow"

def parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parses tool calls from model output and converts them to OpenAI format.
    Handles XML tags, Qwen-style special tokens, and markdown-style JSON blocks.
    """
    tool_calls = []
    clean_text = text
    
    # 1. Handle XML-style tool calls (Qwen format)
    # Example: <tool_call><function=terminal><parameter=command>ls</parameter></function></tool_call>
    xml_pattern = r'<tool_call>(.*?)</tool_call>'
    xml_matches = list(re.finditer(xml_pattern, text, re.DOTALL))
    
    for match in xml_matches:
        content = match.group(1)
        func_match = re.search(r'<function=(.*?)>(.*?)</function>', content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1).strip()
            params_content = func_match.group(2).strip()
            
            parameters = {}
            param_matches = re.findall(r'<parameter=(.*?)>(.*?)</parameter>', params_content, re.DOTALL)
            for p_name, p_value in param_matches:
                parameters[p_name.strip()] = p_value.strip()
            
            # Always include the tool call
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(parameters)
                }
            })
    
    # Remove XML tool calls from clean_text
    clean_text = re.sub(xml_pattern, '', clean_text, flags=re.DOTALL)
    
    # 2. Handle Markdown-style JSON tool calls
    # Example: ```json\n{"tool": "terminal", "parameters": {"command": "ls"}}\n```
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    json_matches = list(re.finditer(json_pattern, clean_text, re.DOTALL))
    
    for match in json_matches:
        try:
            data = json.loads(match.group(1))
            # Check if it looks like a tool call
            if "tool" in data and "parameters" in data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": data["tool"],
                        "arguments": json.dumps(data["parameters"])
                    }
                })
                # Remove from clean_text
                clean_text = clean_text.replace(match.group(0), "")
        except:
            pass
            
    # 3. Handle Qwen-style special token blocks if they appear in text
    # Example: <|tool_call_start|>...
    if "<|tool_call_start|>" in clean_text:
        # This is a bit more complex, usually it's followed by a JSON or specific format
        pass

    # Special handling for 'think' or reasoning
    # We've already added it to tool_calls, but we might want to keep the text 
    # as well if there's no other content, to avoid empty content errors.
    # However, OpenHands usually handles tool_calls with null content.
    pass

    return tool_calls, clean_text.strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    global model_adapter
    if model_adapter is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # print(f"Received request: {request.json()}") # Debugging
    
    # Construct messages list for chat template
    messages_dict = []
    for m in request.messages:
        msg = {"role": m.role}
        content = m.content
        if isinstance(content, list):
            # Extract text from complex content list (LiteLLM/OpenAI format)
            content_text = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content_text += item.get("text", "")
            content = content_text
        
        # Templates often require content to be a string, not None
        msg["content"] = content if content is not None else ""
            
        # Include tool-related fields if present
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        if m.tool_calls:
            # IMPORTANT: Many chat templates (like Qwen) expect tool_calls[i].function.arguments
            # to be a Python dictionary, not a JSON string.
            processed_tool_calls = []
            for tc in m.tool_calls:
                new_tc = tc.copy()
                if "function" in new_tc and "arguments" in new_tc["function"]:
                    args = new_tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            new_tc["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            pass # Keep as string if not valid JSON
                processed_tool_calls.append(new_tc)
            msg["tool_calls"] = processed_tool_calls
            
        messages_dict.append(msg)
    
    # Apply chat template if available
    # We pass the tools to the chat template if supported
    if request.tools and hasattr(model_adapter.tokenizer, "apply_chat_template"):
        try:
            prompt = model_adapter.tokenizer.apply_chat_template(
                messages_dict, 
                tools=request.tools,
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback if tools aren't supported by the specific chat template version
            prompt = model_adapter.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
    elif hasattr(model_adapter.tokenizer, "apply_chat_template") and model_adapter.tokenizer.chat_template:
        prompt = model_adapter.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback simple template
        prompt = ""
        for m in messages_dict:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt += "Assistant:"

    # Tokenize input
    inputs = model_adapter.tokenizer(prompt, return_tensors="pt").to(model_adapter.model.device)
    
    # Prepare sparse attention metadata required by the custom attention layers
    sparse_meta_data = {
        "batch_size": inputs.input_ids.shape[0],
        "num_heads": model_adapter.model.config.num_attention_heads,
        "head_dim": model_adapter.model.config.hidden_size // model_adapter.model.config.num_attention_heads,
        "seq_len": inputs.input_ids.shape[1],
    }
    
    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": request.max_tokens or 4096,
        "temperature": request.temperature if request.temperature > 0 else 0.001,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "do_sample": request.temperature > 0,
        "pad_token_id": model_adapter.tokenizer.pad_token_id,
        "eos_token_id": model_adapter.tokenizer.eos_token_id,
    }
    
    # Generate response
    with torch.no_grad():
        if model_adapter.sparse_attention_config is not None:
            # Run within sparse mode context
            with model_adapter.enable_sparse_mode():
                outputs = model_adapter.model.generate(
                    **inputs,
                    **gen_kwargs,
                    sparse_meta_data=sparse_meta_data
                )
        else:
            # Standard dense execution
            outputs = model_adapter.model.generate(
                **inputs,
                **gen_kwargs,
            )
    
    # Decode newly generated tokens
    generated_text = model_adapter.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False
    )
    
    # Log raw output for debugging
    with open("server_debug.log", "a") as f:
        f.write(f"\n--- NEW REQUEST ---\n")
        f.write(f"PROMPT: {prompt[:500]}...\n")
        f.write(f"RAW OUTPUT: {generated_text}\n")
    
    # Clean up special tokens like <|endoftext|> or <|im_end|> if they appear
    # but keep them for tool call parsing if they are part of the protocol
    
    # Parse tool calls from the generated text
    tool_calls, clean_text = parse_tool_calls(generated_text)
    
    # OpenHands (and some other clients) can fail if BOTH content and tool_calls are present
    # in some versions, but usually content is allowed as "thinking" before tools.
    # However, if content is empty and no tool calls, it MUST be at least an empty string.
    
    # Strip any remaining special tokens from clean_text for the agent
    if model_adapter.tokenizer.eos_token:
        clean_text = clean_text.replace(model_adapter.tokenizer.eos_token, "")
    # Add other common stop tokens if needed
    for stop_token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        clean_text = clean_text.replace(stop_token, "")
    
    clean_text = clean_text.strip()
    
    # Construct OpenAI-compatible response body
    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": clean_text if clean_text else "", # Ensure it's at least empty string
        },
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }
    
    if tool_calls:
        choice["message"]["tool_calls"] = tool_calls
        # If there are tool calls, OpenHands might expect the thinking part in content
        # or it might just follow the tool calls.
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    response_body = {
        "id": completion_id,
        "object": "chat.completion",
        "created": 123456789,
        "model": request.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
            "total_tokens": len(outputs[0]),
        },
    }
    
    return response_body

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models list endpoint.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": model_adapter.model_name if model_adapter else "unknown",
                "object": "model",
                "created": 123456789,
                "owned_by": "sparse-attention-hub",
            }
        ],
    }

def main():
    parser = argparse.ArgumentParser(
        description="Start OpenAI-compatible sparse attention server"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run the model on (e.g. 'cuda:0')"
    )
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument("remaining", nargs="*", help="Port (optionally preceded by 'and')")

    args = parser.parse_args()

    # Parse port from remaining arguments
    if not args.remaining:
        print("Error: Port number is required.")
        sys.exit(1)

    try:
        port = int(args.remaining[-1])
    except ValueError:
        print(f"Error: Invalid port number '{args.remaining[-1]}'.")
        sys.exit(1)

    global model_adapter

    # Use the configuration defined at the top of the file
    sparse_config = SPARSE_CONFIG

    if sparse_config:
        print(f"Running in SPARSE mode with config: {sparse_config}")
    else:
        print("Running in DENSE mode.")

    print(f"Loading model: {args.model}...")
    
    # Device and dtype setup
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    
    try:
        model_adapter = ModelAdapterHF(
            model_name=args.model,
            sparse_attention_config=sparse_config,
            model_kwargs={
                "torch_dtype": dtype,
                "device_map": device,
                "attn_implementation": "sdpa" if "cuda" in device else "eager",
            },
            device=device
        )
        
        # Patch _validate_model_kwargs to an empty function
        # This allows passing sparse_meta_data as a kwarg to generate()
        def _empty_validate_model_kwargs(self, model_kwargs: dict) -> None:
            """Empty function to bypass model kwargs validation."""
            pass

        model_adapter.model._validate_model_kwargs = types.MethodType(
            _empty_validate_model_kwargs, model_adapter.model
        )

        # Device-aware forward wrapper for additional safety
        _orig_forward = model_adapter.model.forward
        model_dev = next(model_adapter.model.parameters()).device

        def _wrapped_forward(*args, **kwargs):
            new_args = tuple(
                a.to(model_dev) if isinstance(a, torch.Tensor) else a for a in args
            )
            new_kwargs = {
                k: v.to(model_dev) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            return _orig_forward(*new_args, **new_kwargs)

        model_adapter.model.forward = _wrapped_forward
        print("Successfully patched model forward and validation for sparse metadata handling.")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

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
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger


import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

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
        SinkMaskerConfig,
        #PQCacheConfig,
        OracleTopKConfig,

    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations.adaptive_sampling import (
        AdaptiveSamplingMaskerConfig,
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


#Sparse Attention ()
SPARSE_CONFIG: Optional[ResearchAttentionConfig] = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        OracleTopKConfig(heavy_size=0.1),
        AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.05,
            epsilon=0.1,
            delta=0.1,
            init_offset=128,
            local_offset=128,
        ),
    ]
)

# ==============================================================================

# Global to track turns for warmup phase
REQUEST_COUNT = 0

app = FastAPI(title="Sparse Attention Model Server")
model_adapter: Optional[ModelAdapterHF] = None

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
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
        pass

    
    pass

    return tool_calls, clean_text.strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    global model_adapter, REQUEST_COUNT
    if model_adapter is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Increment turn counter
    REQUEST_COUNT += 1

    # Reset the dense logging flag for this new request
    if hasattr(model_adapter, 'sparse_attention') and model_adapter.sparse_attention:
        model_adapter.sparse_attention._logged_dense = False

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
        f.write(f"PROMPT: {prompt}\n")  # Full prompt, not truncated
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
                "dtype": dtype,
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

        # Patch sparse attention to use dense fallback for prefill automatically
        if sparse_config:
            from sparse_attention_hub.sparse_attention.utils.kv_utils import _get_num_key_value_groups, repeat_kv
            
            sparse_attn_instance = model_adapter.sparse_attention
            orig_custom_attn = sparse_attn_instance.custom_attention
            
            def patched_custom_attention(self, module, queries, keys, values, attention_mask, scaling, dropout, sparse_meta_data, **kwargs):
                # 1. Check if we should use dense (SDPA)
                # Prefill step (more than 1 query token) OR warmup phase (first 2 turns)
                is_prefill = queries.shape[2] > 1
                use_dense = is_prefill or (REQUEST_COUNT <= 2)

                if use_dense:
                    # Only print once per request, not per layer
                    if not hasattr(self, '_logged_dense') or not getattr(self, '_logged_dense', False):
                        phase = "PREFILL" if is_prefill else f"TURN {REQUEST_COUNT} (WARMUP)"
                        print(f"[{phase}] Using SDPA (Dense) path for {queries.shape[2]} tokens", flush=True)
                        self._logged_dense = True
                        self._was_prefill = is_prefill

                    # Native SDPA requires queries, keys, values to have the same number of heads (GQA/MQA handling)
                    num_key_value_groups = _get_num_key_value_groups(queries, keys)
                    key_states = repeat_kv(keys, num_key_value_groups)
                    value_states = repeat_kv(values, num_key_value_groups)

                    # Compute using SDPA
                    # Note: queries shape is (B, H, L, D)
                    # output shape will be (B, H, L, D)
                    output = F.scaled_dot_product_attention(
                        queries, key_states, value_states,
                        attn_mask=attention_mask,
                        dropout_p=dropout if module.training else 0.0,
                        is_causal=False
                    )

                    # custom_attention expects (B, L, H, D) output (matching get_masked_attention_output)
                    return output.transpose(1, 2).contiguous(), None

                # 2. Verify we are switching to Sparse for decoding (query length 1)
                if queries.shape[2] == 1 and getattr(self, '_was_prefill', False):
                    if kwargs.get('layer_idx') == 0:
                        print(f"[DECODING] Switching to Sparse (Vattn) path for generation...", flush=True)
                        self._was_prefill = False

                # Otherwise use the regular sparse attention ()
                return orig_custom_attn(
                    module, queries, keys, values, attention_mask, scaling, dropout, sparse_meta_data, **kwargs
                )
            
            sparse_attn_instance.custom_attention = types.MethodType(patched_custom_attention, sparse_attn_instance)
            print("Successfully patched sparse attention to use SDPA dense fallback for prefill and warmup.")

        print("Successfully patched model forward and validation for sparse metadata handling.")
        print("Model loaded successfully.")

        
        #metrics logging
        result_dir = Path("./server_metrics/")
        result_dir.mkdir(exist_ok=True)
        metric_logger = MicroMetricLogger()
        metric_logger.configure_logging(
            log_path=result_dir,
            enabled_metrics=[
                "research_attention_density",
                "research_attention_output_error",
            ],
        )
        metric_logger.flush()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Starting server on 0.0.0.0:{port}...")
        
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

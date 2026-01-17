#!/usr/bin/env python3
"""
OpenAI-compatible server for sparse attention models.

Usage:
    python scripts/start_server.py model_name config_path [and] port
"""

import argparse
import json
import os
import re
import sys
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
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from benchmark.raytune.config_builders.utility import (
        deserialize_sparse_config,
        get_all_masker_config_classes,
    )
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print("Ensure you are running from the project root and have all dependencies installed.")
    sys.exit(1)


def flexible_deserialize_sparse_config(data: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Reconstruct ResearchAttentionConfig from JSON data with flexible format support.

    Handles both the RayTune format (type/params) and a flatter format (name/flattened params).

    Args:
        data: Dictionary representation of the config

    Returns:
        ResearchAttentionConfig instance, or None if data is None or invalid
    """
    if data is None:
        return None

    # Try standard RayTune deserialization first
    config = deserialize_sparse_config(data)
    if config is not None:
        return config

    # Fallback to flexible format
    # Expect either 'type': 'ResearchAttentionConfig' or just 'masker_configs'
    if (
        data.get("type") != "ResearchAttentionConfig"
        and data.get("name") != "ResearchAttentionConfig"
        and "masker_configs" not in data
    ):
        # Check if it's the specific OracleTopK style named config
        if data.get("name") != "OracleTopK" and "masker_configs" not in data:
            return None

    # Dynamically discover all available masker config classes
    config_map = get_all_masker_config_classes()

    # Reconstruct masker configs
    masker_configs = []
    for masker_data in data.get("masker_configs", []):
        # Try 'type' then 'name'
        type_name = masker_data.get("type") or masker_data.get("name")
        if not type_name:
            continue

        config_class = config_map.get(type_name)
        if config_class:
            try:
                # Use 'params' if it exists, otherwise use all other keys as params
                if "params" in masker_data:
                    params = masker_data["params"]
                else:
                    # Filter out metadata keys
                    params = {
                        k: v
                        for k, v in masker_data.items()
                        if k not in ["type", "name"]
                    }
                masker_configs.append(config_class(**params))
            except Exception as e:
                print(f"Warning: Failed to create {type_name}: {e}")
                continue

    if not masker_configs:
        return None

    return ResearchAttentionConfig(masker_configs=masker_configs)


# Predefined configurations that can be used by name instead of a file path
PREDEFINED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "oracle_50": {
        "name": "OracleTopK",
        "masker_configs": [
            {"name": "SinkMaskerConfig", "sink_size": 128},
            {"name": "LocalMaskerConfig", "window_size": 128},
            {"name": "OracleTopKMaskerConfig", "heavy_size": 0.5},
        ],
    },
    "streaming_llm": {
        "name": "ResearchAttentionConfig",
        "masker_configs": [
            {"name": "SinkMaskerConfig", "sink_size": 4},
            {"name": "LocalMaskerConfig", "window_size": 64},
        ],
    },
}

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
                    sparse_meta_data=sparse_meta_data,
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
    parser = argparse.ArgumentParser(description="Start OpenAI-compatible sparse attention server")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g. 'cuda:0')")
    parser.add_argument("model", type=str, help="HuggingFace model name")
    parser.add_argument("config", type=str, help="Path to sparse attention config JSON (or 'dense')")
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
    
    # Load sparse attention configuration
    sparse_config = None
    if args.config.lower() != "dense":
        config_data = None

        # 1. Check if it's a predefined config name
        if args.config.lower() in PREDEFINED_CONFIGS:
            print(f"Using predefined sparse config: {args.config.lower()}")
            config_data = PREDEFINED_CONFIGS[args.config.lower()]
        # 2. Check if it's a JSON string
        elif args.config.strip().startswith("{") and args.config.strip().endswith("}"):
            try:
                config_data = json.loads(args.config)
                print("Using sparse config from JSON string.")
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON string provided for config: {e}")
                sys.exit(1)
        # 3. Check if it's a file path
        elif os.path.exists(args.config):
            with open(args.config, "r") as f:
                try:
                    config_data = json.load(f)
                    print(f"Using sparse config from file: {args.config}")
                except json.JSONDecodeError:
                    print(f"Error: {args.config} is not a valid JSON file.")
                    sys.exit(1)
        else:
            print(
                f"Error: Config not found. Must be 'dense', a predefined name ({', '.join(PREDEFINED_CONFIGS.keys())}), a JSON string, or a valid file path."
            )
            sys.exit(1)

        if config_data:
            sparse_config = flexible_deserialize_sparse_config(config_data)
            if sparse_config is None:
                print(
                    f"Warning: Could not deserialize sparse config from {args.config}. Falling back to DENSE."
                )
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
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

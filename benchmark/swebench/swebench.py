"""SWE-bench benchmark (API-based evaluation)."""

import re
import os
import subprocess
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import sys

from benchmark.base import Benchmark
from benchmark.benchmark_registry import register_benchmark
from sparse_attention_hub.adapters.base import Request, RequestResponse, ModelAdapter

PATCH_EXAMPLE = """--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -17,6 +17,7 @@
 from .core import Model, ModelDefinitionError
 from .mappings import Mapping
 from astropy.modeling.core import CompoundModel
+import numpy as np
 
 __all__ = ["is_separable", "separability_matrix"]
"""

@register_benchmark("swebench")
class SWEBench(Benchmark):
    benchmark_name: str = "swebench"
    all_datasets: List[str] = [
        "swe-bench-m", "swe-bench_lite", "swe-bench_verified",
        "swe-bench_lite_oracle", "swe-bench_lite_bm25"
    ]
    huggingface_dataset_id: str = "princeton-nlp/SWE-bench_Lite" # Default to regular Lite

    dataset_mapping = {
        "swe-bench-m": "princeton-nlp/SWE-bench",
        "swe-bench_lite": "princeton-nlp/SWE-bench_Lite",
        "swe-bench_verified": "princeton-nlp/SWE-bench_Verified",
        "swe-bench_lite_oracle": "princeton-nlp/SWE-bench_Lite_oracle",
        "swe-bench_lite_bm25": "princeton-nlp/SWE-bench_Lite_bm25_27K"
    }

    def __init__(self, subsets_to_run: Optional[List[str]] = None) -> None:
        """Initialize SWE-bench with specific subset."""
        if subsets_to_run is None:
            subsets_to_run = ["swe-bench_lite"]  # Default to regular Lite
        
        # Set HF dataset ID based on requested subset
        # Strip slicing info if present (e.g., swe-bench_lite:0:20 -> swe-bench_lite)
        primary_subset = subsets_to_run[0].split(":")[0]
        self.huggingface_dataset_id = self.dataset_mapping.get(primary_subset, "princeton-nlp/SWE-bench_Lite")
        
        # We call super().__init__ but we need to bypass strict validation if slicing is used
        # So we'll validate manually or modify _validate_subsets
        super().__init__(subsets_to_run=subsets_to_run)
        
        # Fix "dubious ownership" error in Docker environments
        try:
            subprocess.run("git config --global --add safe.directory '*'", shell=True, check=True)
        except Exception as e:
            print(f"Warning: Failed to set git safe directory: {e}")

        # Directory to cache repositories
        self.repo_cache_dir = os.path.expanduser("~/swe-bench-repos")
        if not os.path.exists(self.repo_cache_dir):
            os.makedirs(self.repo_cache_dir)

    def _checkout_repo(self, repo: str, base_commit: str) -> str:
        """Checkout a repo at a specific commit into a temporary directory."""
        repo_name = repo.replace("/", "__")
        cached_repo = os.path.join(self.repo_cache_dir, repo_name)
        
        if not os.path.exists(cached_repo):
            print(f"Cloning {repo} to {cached_repo}...")
            subprocess.run(f"git clone https://github.com/{repo} {cached_repo}", shell=True, check=True)
        
        # Create a temporary workspace for this instance
        workspace_dir = tempfile.mkdtemp(prefix=f"swe_agent_{repo_name}_")
        
        # Regular clone (removed --local to support cross-device volume mounts in Docker)
        subprocess.run(f"git clone {cached_repo} {workspace_dir}", shell=True, check=True)
        
        # Checkout the base commit
        subprocess.run(f"cd {workspace_dir} && git reset --hard {base_commit}", shell=True, check=True)
        
        return workspace_dir

    def _get_agent_system_prompt(self) -> str:
        return (
            "You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.\n\n"
            "Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n"
            "Include a THOUGHT section before your command where you explain your reasoning process.\n"
            "Format your response as shown in <format_example>.\n\n"
            "<format_example>\n"
            "THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n"
            "```bash\n"
            "ls -la\n"
            "```\n"
            "</format_example>\n"
            "Failure to follow these rules will cause your response to be rejected."
        )

    def _get_agent_instructions(self, problem_statement: str) -> str:
        return (
            "# Task Instructions\n\n"
            "## Overview\n"
            "You're a software engineer interacting continuously with a computer by submitting commands.\n"
            "You'll be helping implement necessary changes to meet requirements in the PR description.\n"
            "Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.\n\n"
            "IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.\n\n"
            "## PR Description\n"
            f"<pr_description>\n{problem_statement}\n</pr_description>\n\n"
            "## Important Boundaries\n"
            "- MODIFY: Regular source code files\n"
            "- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)\n\n"
            "## Recommended Workflow\n"
            "1. Analyze the codebase by finding and reading relevant files\n"
            "2. Create a script to reproduce the issue\n"
            "3. Edit the source code to resolve the issue\n"
            "4. Verify your fix works by running your script again\n"
            "5. Test edge cases to ensure your fix is robust\n\n"
            "## Command Execution Rules\n"
            "1. You write a single command\n"
            "2. The system executes that command in a subshell\n"
            "3. You see the result\n"
            "4. You write your next command\n\n"
            "## Submission\n"
            "When you've completed your changes or can't make further progress, issue exactly the following command to stop working and submit your changes:\n"
            "```bash\n"
            "echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached\n"
            "```\n"
            "This command will submit your changes. You cannot continue working on this task after submitting.\n"
        )

    def _is_safe_command(self, cmd: str) -> bool:
        """Basic safety check for commands."""
        forbidden = [
            "rm -rf /", "rm -rf ~", "rm -rf $HOME",
            "chmod -R 777", "chown", "passwd",
            "curl ", "wget ", "ssh ", "nc ", "netcat "
        ]
        for f in forbidden:
            if f in cmd:
                return False
        return True

    def _extract_bash_command(self, model_output: str) -> Optional[str]:
        """Extract the bash command from the model output."""
        match = re.search(r"```bash\n(.*?)\n```", model_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback for some models that omit \n
        match = re.search(r"```bash(.*?)\n```", model_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _make_code_text(self, files_dict: Any) -> str:
        """Helper to format file contents into a string."""
        if not files_dict or not isinstance(files_dict, dict):
            return ""
        text = ""
        for path, content in files_dict.items():
            text += f"--- {path}\n{content}\n\n"
        return text

    def _run_agent_loop(
        self, 
        instance: pd.Series, 
        adapter: ModelAdapter, 
        generation_kwargs: Dict[str, Any], 
        request_kwargs: Dict[str, Any],
        max_turns: int = 30
    ) -> str:
        """Run an interactive agent loop for a single SWE-bench instance."""
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem_statement = instance["problem_statement"]
        
        print(f"[{instance_id}] Starting agent loop...")
        
        workspace_dir = None
        try:
            workspace_dir = self._checkout_repo(repo, base_commit)
            
            # Initial history
            history = f"{self._get_agent_system_prompt()}\n\n"
            history += self._get_agent_instructions(problem_statement)
            
            for turn in range(max_turns):
                print(f"[{instance_id}] Turn {turn+1}/{max_turns}")
                
                # In zero-shot adapter, we just pass the history as context
                request = Request(context=history, questions=["What is your next command?"], answer_prefix="THOUGHT:")
                
                # Get model response
                response: RequestResponse = adapter.process_request(request, generation_kwargs, request_kwargs)
                raw_response = response.responses if isinstance(response.responses, str) else response.responses[0]
                model_output = "THOUGHT: " + raw_response
                
                print(f"[{instance_id}] Model response length: {len(model_output)}")
                
                # Extract command
                cmd = self._extract_bash_command(model_output)
                if not cmd:
                    print(f"[{instance_id}] Error: No bash command found.")
                    history += f"\n{model_output}\n\nError: No bash code block found. Remember to wrap your command in ```bash ... ```"
                    continue
                
                # Safety check
                if not self._is_safe_command(cmd):
                    print(f"[{instance_id}] Blocked unsafe command: {cmd}")
                    history += f"\n{model_output}\n\nError: Command was blocked for safety reasons."
                    continue

                print(f"[{instance_id}] Executing: {cmd}")
                
                # Check for submission
                if "MINI_SWE_AGENT_FINAL_OUTPUT" in cmd:
                    print(f"[{instance_id}] Submission detected!")
                    result = subprocess.run(f"cd {workspace_dir} && {cmd}", shell=True, capture_output=True, text=True)
                    patch = result.stdout.split("MINI_SWE_AGENT_FINAL_OUTPUT")[-1].strip()
                    return patch

                # Execute command
                try:
                    # Run in a subshell inside the workspace
                    result = subprocess.run(f"cd {workspace_dir} && {cmd}", shell=True, capture_output=True, text=True, timeout=120)
                    output = result.stdout + result.stderr
                    if not output.strip():
                        output = "(no output)"
                except subprocess.TimeoutExpired:
                    output = "Error: Command timed out after 120 seconds."
                except Exception as e:
                    output = f"Error during execution: {str(e)}"
                
                # Update history
                history += f"\n{model_output}\n\nExit code: {result.returncode if 'result' in locals() else 'unknown'}\n{output}\n"
                
                # Context management (sliding window)
                if len(history) > 60000:
                    history = history[:4000] + "\n\n... (history truncated) ...\n\n" + history[-50000:]

            print(f"[{instance_id}] Reached max turns.")
            result = subprocess.run(f"cd {workspace_dir} && git add -A && git diff --cached", shell=True, capture_output=True, text=True)
            return result.stdout.strip()
            
        except Exception as e:
            print(f"[{instance_id}] Agent loop failed: {str(e)}")
            return ""
        finally:
            if workspace_dir and os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)

    def _process_all_requests(
        self, 
        adapter: ModelAdapter, 
        dataset_df: pd.DataFrame,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process all samples through an agentic loop if regular dataset is used."""
        import json
        
        # Add model info for incremental saves
        dataset_df["model_name_or_path"] = getattr(adapter, "model_name", "unknown")
        
        # Detect if we should use agentic mode
        is_oracle = "oracle" in self.huggingface_dataset_id or "bm25" in self.huggingface_dataset_id
        
        if is_oracle:
            return super()._process_all_requests(adapter, dataset_df, generation_kwargs, request_kwargs)
        
        max_requests = request_kwargs.get("max_requests", sys.maxsize)
        dataset_df = dataset_df.head(max_requests).copy()
        dataset_df["predicted_answer"] = None
        
        result_dir = request_kwargs.get("result_dir")
        max_turns = request_kwargs.get("max_turns", 30)
        
        for idx, row in dataset_df.iterrows():
            patch = self._run_agent_loop(row, adapter, generation_kwargs, request_kwargs, max_turns=max_turns)
            dataset_df.at[idx, "predicted_answer"] = patch
            
            # Incremental save after each sample
            if result_dir:
                try:
                    results_so_far = dataset_df.dropna(subset=["predicted_answer"])
                    completed_instances = self.post_run_evaluate(results_so_far)
                    
                    # Also include intermediate metrics for visibility
                    intermediate_path = os.path.join(result_dir, "intermediate_metrics.json")
                    with open(intermediate_path, "w") as f:
                        json.dump(completed_instances, f, indent=2)
                    
                    # Periodically print progress
                    if len(completed_instances) % 1 == 0:
                        print(f"\n[Incremental Save] Processed {len(completed_instances)} samples. Saved to {intermediate_path}")
                except Exception as e:
                    print(f"Warning: Failed to save intermediate results: {e}")
            
        return dataset_df

    def _validate_subsets(self, subsets: List[str]) -> None:
        """Validate that requested subsets exist in all_datasets (supporting slicing)."""
        for subset in subsets:
            base_subset = subset.split(":")[0]
            if base_subset not in self.all_datasets:
                raise ValueError(
                    f"Invalid subset: {subset}. Base subset {base_subset} not in {self.all_datasets}"
                )

    def _load_datasets(self) -> pd.DataFrame:
        from datasets import load_dataset

        ds = load_dataset(self.huggingface_dataset_id, split="test")
        df = ds.to_pandas()

        # Handle slicing if specified in the subset name (e.g., "swe-bench_lite:0:20")
        subset_name = self.subsets_to_run[0]
        if ":" in subset_name:
            parts = subset_name.split(":")
            if len(parts) == 3:
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    print(f"Slicing dataset: {start} to {end}")
                    df = df.iloc[start:end].copy()
                except ValueError:
                    print(f"Warning: Could not parse slice indices from {subset_name}")

        is_oracle = "oracle" in self.huggingface_dataset_id or "bm25" in self.huggingface_dataset_id
        
        if is_oracle:
            df["context"] = df.apply(self._prompt_style_3, axis=1)
            df["question"] = ""
            df["answer_prefix"] = "<patch>\n--- a/"
        else:
            df["context"] = df["problem_statement"]
            df["question"] = "Agentic Loop Target"
            df["answer_prefix"] = ""
            
        df["task"] = self.subsets_to_run[0]
        df["max_new_tokens"] = 8192
        df["context_length"] = None

        return df

    def _extract_patch(self, model_output: str) -> str:
        """Extracts the diff/patch from the model response."""
        if model_output.startswith("diff --git") or model_output.startswith("--- a/"):
            return model_output
            
        return self._extract_patch_zero_shot(model_output)

    def _extract_patch_zero_shot(self, model_output: str) -> str:
        """Standard zero-shot extraction logic."""
        if not model_output:
            return ""
            
        if not model_output.startswith("--- a/") and not model_output.startswith("<patch>"):
            model_output = "--- a/" + model_output

        patch_match = re.search(r"<patch>(.*?)</patch>", model_output, re.DOTALL)
        if patch_match:
            clean = patch_match.group(1).strip()
        elif "--- a/" in model_output and "+++ b/" in model_output:
            code_match = re.search(r"```(?:\w+)?\n(.*?)\n```", model_output, re.DOTALL)
            if code_match:
                clean = code_match.group(1).strip()
            else:
                clean = model_output.replace("<patch>", "").replace("</patch>", "").strip()
        else:
            clean = model_output.strip()

        lines = clean.split("\n")
        filtered_lines = []
        for line in lines:
            if line.strip().startswith("#...") or line.strip().startswith("...") or "# (rest of" in line:
                continue
            filtered_lines.append(line)
        
        return "\n".join(filtered_lines)

    def _make_code_text(self, files_dict: Any) -> str:
        """Helper to format file contents into a string."""
        if not files_dict or not isinstance(files_dict, dict):
            return ""
        text = ""
        for path, content in files_dict.items():
            text += f"--- {path}\n{content}\n\n"
        return text

    def _prompt_style_3(self, instance: pd.Series) -> str:
        """Comprehensive prompt style with strict diff instructions."""
        if "text" in instance and instance["text"]:
            base_prompt = str(instance["text"])
        else:
            premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
            readmes_text = self._make_code_text(instance.get("readmes", {}))
            code_text = self._make_code_text(instance.get("file_contents", {}))
            problem_statement = instance.get("problem_statement", "")
            base_prompt = f"{premise}\n<issue>\n{problem_statement}\n</issue>\n<code>\n{readmes_text}\n{code_text}\n</code>"

        instructions = (
            "\nI need you to solve the provided issue by generating a single patch file that I can apply "
            "directly to this repository using git apply. \n"
            "CRITICAL RULES:\n"
            "1. Only respond with the patch file content wrapped in <patch> tags.\n"
            "2. Do NOT use ellipses (e.g., #...) or summaries inside the patch.\n"
            "3. Ensure the hunk headers (@@ -L,l +L,l @@) are accurate.\n"
            "4. Match the surrounding context lines exactly.\n"
            "5. Use the following format for the patch:\n"
            "<patch>\n"
            f"{PATCH_EXAMPLE}\n"
            "</patch>\n"
            "Respond below:"
        )
        
        return base_prompt + instructions

    def post_run_evaluate(self, results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare SWE-bench submissions in sb-cli list format."""
        predictions = []

        for _, row in results_df.iterrows():
            raw_output = row.get("predicted_answer", "")
            patch = self._extract_patch(raw_output)
            
            predictions.append({
                "instance_id": row.get("instance_id"),
                "model_patch": patch,
                "model_name_or_path": row.get("model_name_or_path", "unknown_model")
            })

        return predictions

    def post_run_submit(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """DEPRECATED: Use post_run_evaluate for sb-cli format."""
        return {"submissions": self.post_run_evaluate(results_df)}

#!/usr/bin/env python3
"""
Re-sync SWE-bench model_patch fields from mini-swe-agent trajectory JSON.

mini sometimes writes preds.json patches from assistant prose (or mid-file
fragments). Prefer, in order: (1) the final ``role: "exit"`` message
(``extra.submission`` / ``content``), which matches what mini submits; (2) the
last tool ``raw_output`` containing ``diff --git`` (covers ``cat patch.txt``
after ``git diff ... > patch.txt``, where stdout was empty); (3) bash results
for commands containing ``git diff`` (stdout not redirected).

Usage:
    python scripts/repair_mini_preds_from_trajs.py \\
        --preds benchmarks/mini/runs/my_run/preds.json \\
        --output-dir benchmarks/mini/runs/my_run \\
        [--write path/to/out.json]   # default: overwrite --preds

    # Replace every patch that has a traj, using last git diff (overrides good patches):
    python scripts/repair_mini_preds_from_trajs.py ... --force-all-from-traj
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_valid_unified_diff(patch: str) -> bool:
    """True if patch looks like a git unified diff SWE-bench can apply."""
    p = patch.strip()
    if not p:
        return True  # empty patch is valid (no-op submission)
    if not p.startswith("diff --git "):
        return False
    return "--- " in p and "+++ " in p


def _strip_mini_exit_prefix(text: str) -> str:
    """Remove leading exit-code line mini often prepends to bash output."""
    t = text
    if re.match(r"^\d+\n", t):
        t = t[t.index("\n") + 1 :]
    return t


def _normalize_patch_from_text(text: str) -> str | None:
    """If *text* embeds a valid unified diff, return it; else None."""
    if not text or not isinstance(text, str):
        return None
    t = _strip_mini_exit_prefix(text)
    if "diff --git" not in t:
        return None
    idx = t.index("diff --git")
    patch = t[idx:].rstrip()
    if not is_valid_unified_diff(patch):
        return None
    return patch + ("\n" if not patch.endswith("\n") else "")


def patch_from_exit_message(messages: list[dict[str, Any]]) -> str | None:
    """
    mini-swe-agent 1.1 ends with role ``exit`` and the submitted patch in
    ``extra.submission`` or ``content``.

    None  -> no exit message or non-empty body without a diff (try other methods).
    ""    -> explicit empty submission.
    str   -> unified diff.
    """
    for msg in reversed(messages):
        if msg.get("role") != "exit":
            continue
        extra = msg.get("extra") or {}
        sub = extra.get("submission")
        if sub is None:
            c = msg.get("content")
            sub = c if isinstance(c, str) else None
        if not isinstance(sub, str):
            return None
        if not sub.strip():
            return ""
        patch = _normalize_patch_from_text(sub)
        return patch if patch is not None else None
    return None


def last_diff_from_tool_outputs(messages: list[dict[str, Any]]) -> str | None:
    """Last tool message whose stdout contains a unified diff (e.g. cat patch.txt)."""
    last: str | None = None
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        raw = msg.get("extra", {}).get("raw_output")
        if raw is None:
            c = msg.get("content")
            raw = c if isinstance(c, str) else ""
        patch = _normalize_patch_from_text(raw)
        if patch is not None:
            last = patch
    return last


def _bash_command_for_tool_call(messages: list[dict[str, Any]], tool_call_id: str) -> str | None:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc.get("id") != tool_call_id:
                continue
            fn = tc.get("function") or {}
            if fn.get("name") != "bash":
                return None
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                return None
            cmd = args.get("command")
            return cmd if isinstance(cmd, str) else None
    return None


def last_git_diff_patch_from_traj(messages: list[dict[str, Any]]) -> str | None:
    """
    Return the patch implied by the last `git diff` bash result in the traj.

    None  -> no git diff tool call appeared (caller should not overwrite).
    ""    -> last git diff was empty (clean working tree).
    str   -> unified diff text.
    """
    saw_git_diff = False
    last: str | None = None

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tid = msg.get("tool_call_id")
        if not tid:
            continue
        cmd = _bash_command_for_tool_call(messages, tid)
        if not cmd or "git diff" not in cmd:
            continue
        saw_git_diff = True

        raw = msg.get("extra", {}).get("raw_output")
        if raw is None:
            c = msg.get("content")
            raw = c if isinstance(c, str) else ""
        if not isinstance(raw, str):
            continue

        normalized = _strip_mini_exit_prefix(raw)
        if "diff --git" in normalized:
            idx = normalized.index("diff --git")
            patch = normalized[idx:].rstrip()
            if is_valid_unified_diff(patch):
                last = patch + ("\n" if not patch.endswith("\n") else "")
            else:
                last = ""
        else:
            last = ""

    if not saw_git_diff:
        return None
    return last if last is not None else ""


def extract_patch_from_traj(messages: list[dict[str, Any]]) -> str | None:
    """
    Best-effort patch for SWE-bench: exit submission, then tool diffs, then git diff commands.
    None means we could not infer a patch (caller should not overwrite).
    """
    pe = patch_from_exit_message(messages)
    if pe is not None:
        return pe
    pl = last_diff_from_tool_outputs(messages)
    if pl is not None:
        return pl
    return last_git_diff_patch_from_traj(messages)


def find_traj_path(run_output_dir: Path, instance_id: str) -> Path | None:
    for replica in sorted(run_output_dir.glob("replica_*")):
        candidate = replica / instance_id / f"{instance_id}.traj.json"
        if candidate.is_file():
            return candidate
    return None


def repair_preds(
    preds: dict[str, Any],
    run_output_dir: Path,
    *,
    only_if_invalid: bool,
) -> tuple[int, int, list[str]]:
    """
    Mutate preds in place. Returns (n_updated, n_instances, list of instance ids updated).
    """
    updated: list[str] = []
    for instance_id, entry in preds.items():
        if not isinstance(entry, dict):
            continue
        old = entry.get("model_patch", "")
        if not isinstance(old, str):
            old = ""
        if only_if_invalid and is_valid_unified_diff(old):
            continue

        traj_path = find_traj_path(run_output_dir, instance_id)
        if traj_path is None:
            continue

        try:
            traj = json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        messages = traj.get("messages", [])
        if not isinstance(messages, list):
            continue

        new_patch = extract_patch_from_traj(messages)
        if new_patch is None:
            continue
        if new_patch == old:
            continue

        entry["model_patch"] = new_patch
        updated.append(instance_id)

    return len(updated), len(preds), updated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True, help="preds.json path")
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory containing replica_*/<instance_id>/*.traj.json",
    )
    ap.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Output path (default: overwrite --preds)",
    )
    ap.add_argument(
        "--force-all-from-traj",
        action="store_true",
        help="Replace every patch that has a traj, using last git diff (default: only broken patches)",
    )
    args = ap.parse_args()

    pred_path = args.preds if args.preds.is_absolute() else _project_root() / args.preds
    out_dir = args.output_dir if args.output_dir.is_absolute() else _project_root() / args.output_dir
    write_path = args.write
    if write_path is not None:
        write_path = write_path if write_path.is_absolute() else _project_root() / write_path
    else:
        write_path = pred_path

    if not pred_path.is_file():
        print(f"Missing preds file: {pred_path}", file=sys.stderr)
        sys.exit(1)
    if not out_dir.is_dir():
        print(f"Missing output dir: {out_dir}", file=sys.stderr)
        sys.exit(1)

    preds = json.loads(pred_path.read_text())
    if not isinstance(preds, dict):
        print("preds.json must be a JSON object", file=sys.stderr)
        sys.exit(1)

    only_if_invalid = not args.force_all_from_traj
    n_updated, n_total, ids = repair_preds(preds, out_dir, only_if_invalid=only_if_invalid)
    write_path.write_text(json.dumps(preds, indent=2))
    print(f"Updated {n_updated}/{n_total} instances → {write_path}")
    if ids:
        print("Patched instance_ids:")
        for iid in sorted(ids)[:40]:
            print(f"  {iid}")
        if len(ids) > 40:
            print(f"  ... and {len(ids) - 40} more")


if __name__ == "__main__":
    main()

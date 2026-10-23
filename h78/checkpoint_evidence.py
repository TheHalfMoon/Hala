"""H78 host-side admission of an actually finished optimizer checkpoint.

Uses existing full artifact checks, never runs or imports CUDA/model weights.
"""
from __future__ import annotations

import math
from pathlib import Path

from training.checkpoints import load_checkpoint
from training.artifacts import strict_json

SCHEMA = "hala.h78.completed-host-checkpoint.v1"


def checkpoint_sha_if_present(host_checkpoint_dir: Path) -> str | None:
    """Validate and fingerprint an existing host checkpoint before launching."""
    root = Path(host_checkpoint_dir)
    if not (root / "latest.json").exists():
        return None
    # No source may inherit an invalid host pointer as success evidence.
    load_checkpoint(root, None)
    pointer = strict_json((root / "latest.json").read_text("utf8"))
    return pointer["checkpoint_sha256"]


def qualify_completed_host_checkpoint(host_checkpoint_dir: Path, expected_steps: int,
                                      before_sha: str | None) -> dict:
    """Complete training evidence must match the requested step budget.

    A valid old checkpoint (or an incomplete earlier step) is valuable for
    salvage but does not prove a *new*, completed training run succeeded.
    """
    if type(expected_steps) is not int or not 1 <= expected_steps <= 600:
        raise ValueError("bounded requested optimizer step count required")
    root = Path(host_checkpoint_dir)
    version, state = load_checkpoint(root, None)
    pointer = strict_json((root / "latest.json").read_text("utf8"))
    sha = pointer["checkpoint_sha256"]
    if before_sha is not None and sha == before_sha:
        raise ValueError("stale preexisting checkpoint is not a completed new run")
    if (state["step"] != expected_steps or state["max_steps"] != expected_steps
            or state["changed_parameter_tensors"] <= 0):
        raise ValueError("host checkpoint does not prove complete gradient-bearing steps")
    steps = state["runtime"].get("qualification_steps")
    if not isinstance(steps, list) or len(steps) != expected_steps:
        raise ValueError("full real optimizer-step qualification history required")
    for i, info in enumerate(steps, start=1):
        if (not isinstance(info, dict) or type(info.get("step")) is not int
                or info["step"] != i
                or type(info.get("changed_parameter_tensors_this_step")) is not int
                or info["changed_parameter_tensors_this_step"] <= 0
                or type(info.get("gradient_norm_before_clip")) not in (float, int)
                or not math.isfinite(info["gradient_norm_before_clip"])
                or info["gradient_norm_before_clip"] <= 0):
            raise ValueError("real updated-gradient receipt missing at an optimizer step")
    return {
        "schema": SCHEMA,
        "status": "VERIFIED_COMPLETED_HOST_CHECKPOINT",
        "step": state["step"],
        "max_steps": state["max_steps"],
        "changed_parameter_tensors": state["changed_parameter_tensors"],
        "dataset_sha256": state["dataset_sha256"],
        "config_sha256": state["config_sha256"],
        "checkpoint_sha256": sha,
        "checkpoint_version": version.relative_to(root.resolve()).as_posix(),
        "private_provider_readback_verified": False,
        "evaluation_required": True,
        "adapter_qualified": False,
    }

"""H82: restore an immutable *GPU-trained* checkpoint after pointer-publish crash.

Never execute weights. Independently verify original full adapter/optimizer
artifact SHA-256 and run identity before atomically publishing a lost pointer.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from training.checkpoints import (atomic_json, load_checkpoint, validate_version)
from training.artifacts import sha256_file

VERSION = re.compile(r"^step-([0-9]{8})-[0-9a-f]{32}$")
INCOMPLETE = re.compile(r"^\.staging-[0-9a-f]{32}$")
SCHEMA = "hala.h82.orphaned-remote-checkpoint.v1"


def recover_orphaned_checkpoint(output: Path, expected: dict) -> dict:
    """Advance latest.json by precisely one validated optimizer step or refuse.

    A checkpoint with no pointer (first optimizer update) is recoverable too.
    Never adopt two competing next-step versions or skip an unverified step.
    """
    output = Path(output)
    if output.is_symlink():
        raise ValueError("linked optimizer output forbidden")
    pointer = output / "latest.json"
    if pointer.is_symlink():
        raise ValueError("linked training checkpoint pointer forbidden")
    current, step = None, 0
    if pointer.exists():
        version, state = load_checkpoint(output, expected)
        current = version.name
        step = state["step"]
    versions = output / "checkpoints"
    if versions.is_symlink():
        raise ValueError("linked training checkpoint versions forbidden")
    if not versions.is_dir():
        if current is not None:
            raise ValueError("missing checkpoint versions directory")
        raise ValueError("no valid optimizer checkpoint to resume")
    candidates = []
    for item in versions.iterdir():
        if item.is_symlink():
            raise ValueError("linked checkpoint version forbidden")
        if INCOMPLETE.fullmatch(item.name):
            if not item.is_dir():
                raise ValueError("invalid interrupted checkpoint staging")
            continue  # only the atomic named version can be published
        parsed = VERSION.fullmatch(item.name)
        if parsed is None or not item.is_dir():
            raise ValueError("unrecognized checkpoint version entry")
        version_step = int(parsed.group(1))
        if version_step < step:
            continue  # immutable older versions from the same experiment
        if version_step == step:
            if item.name != current:
                raise ValueError("conflicting versions at current optimizer step")
            continue
        if version_step != step + 1:
            raise ValueError("noncontiguous unpublished optimizer progress")
        validated, artifacts = validate_version(item, expected)
        if validated["step"] != version_step:
            raise ValueError("unpublished optimizer version step mismatch")
        receipts = validated["runtime"].get("qualification_steps")
        if (validated["changed_parameter_tensors"] <= 0
                or not isinstance(receipts, list)
                or len(receipts) != version_step):
            raise ValueError("unpublished optimizer update evidence incomplete")
        for index, row in enumerate(receipts, start=1):
            if (not isinstance(row, dict)
                    or type(row.get("step")) is not int or row["step"] != index
                    or type(row.get("changed_parameter_tensors_this_step")) is not int
                    or row["changed_parameter_tensors_this_step"] <= 0
                    or type(row.get("gradient_norm_before_clip")) not in (int, float)
                    or not math.isfinite(row["gradient_norm_before_clip"])
                    or row["gradient_norm_before_clip"] <= 0):
                raise ValueError("unpublished optimizer gradient evidence invalid")
        candidates.append(item)
    if len(candidates) > 1:
        raise ValueError("ambiguous unpublished optimizer fork at same step")
    if not candidates:
        if current is None:
            raise ValueError("no complete published or recoverable optimizer checkpoint")
        return {"schema": SCHEMA, "status": "ALREADY_PUBLISHED",
                "step": step, "new_optimizer_step_executed": False}
    orphan = candidates[0]
    proof = sha256_file(orphan / "checkpoint.json")
    atomic_json(pointer, {
        "schema": "hala.training.latest.v1",
        "version": "checkpoints/" + orphan.name,
        "checkpoint_sha256": proof,
    })
    verified, state = load_checkpoint(output, expected)
    if verified != orphan.resolve() or state["step"] != step + 1:
        raise ValueError("recovered optimizer latest pointer did not independently verify")
    return {"schema": SCHEMA, "status": "RECOVERED_ORPHANED_OPTIMIZER_CHECKPOINT",
            "step": state["step"], "checkpoint_sha256": proof,
            "new_optimizer_step_executed": False,
            "private_provider_readback_verified": False,
            "adapter_qualified": False}

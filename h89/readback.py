"""H89: verify a DOWNLOADED private Kaggle adapter+optimizer checkpoint.

CPU-only; do not equate local file SHA checks with independent remote
provider authentication, a model quality evaluation, or commercial readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from h73.kaggle_h73_cpu import TRAIN_SHA
from training.artifacts import sha256_file, strict_json
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

SOURCE_SHA = "5a8396881b655736277b25255d7101f6c602213e"
SCHEMA = "hala.h89.downloaded-kaggle-checkpoint-admission.v1"
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def read_json(path, budget=4 * 1024 * 1024):
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= budget):
        raise ValueError("missing/oversized downloaded Kaggle receipt")
    obj = strict_json(path.read_text("utf8"))
    if not isinstance(obj, dict):
        raise ValueError("JSON object required")
    return obj


def sha(value):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError("exact SHA256 required")
    return value


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def admit_downloaded_output(directory):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("downloaded private Kaggle directory required")
    attempt = read_json(root / "hala-h88-research-attempt.json")
    receipt = read_json(root / "hala-h87-training-receipt.json")
    if (attempt.get("schema") != "hala.h88.noncommercial-private-kaggle-research-attempt.v1"
            or attempt.get("status") != "REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED"
            or attempt.get("stage") != "CHECKPOINT_RELOADED"
            or attempt.get("source_sha") != SOURCE_SHA
            or attempt.get("model_trained_verified") is not True
            or attempt.get("provider_private_readback_verified") is not False
            or attempt.get("commercial_use") is not False
            or type(attempt.get("optimizer_step")) is not int
            or attempt["optimizer_step"] != 1):
        raise ValueError("H88 attempt is not a qualified one-step research run")
    if (receipt.get("schema") != "hala.h87.noncommercial-kaggle-real-qwen-step.v1"
            or receipt.get("status") != "REAL_QWEN_LORA_STEP_CHECKPOINT_VERIFIED"
            or receipt.get("model_id") != MODEL_ID
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("dataset_sha256") != TRAIN_SHA
            or receipt.get("commercial_use") is not False
            or receipt.get("model_quality_evaluated") is not False
            or receipt.get("provider_readback_verified") is not False
            or type(receipt.get("step")) is not int
            or receipt["step"] != 1):
        raise ValueError("H87 TRAIN source/model/step not qualified")
    output = root / "hala-h88-research-checkpoint"
    if output.is_symlink() or not output.is_dir():
        raise ValueError("missing adapter-plus-optimizer checkpoint")
    # Reuse H61 original immutable pointer, metadata SHA, complete member
    # inventory and optimizer artifact validation. No torch/model required.
    version, state = load_checkpoint(output, expected=None)
    pointer = read_json(output / "latest.json", 1024)
    metadata_sha = sha256_file(version / "checkpoint.json")
    if (metadata_sha != sha(pointer.get("checkpoint_sha256"))
            or metadata_sha != sha(attempt.get("checkpoint_sha256"))
            or metadata_sha != sha(receipt.get("checkpoint_sha256"))
            or version.name != receipt.get("checkpoint_version")
            or state.get("step") != 1 or state.get("max_steps") != 1
            or state.get("dataset_sha256") != TRAIN_SHA):
        raise ValueError("checkpoint does not match pinned H87/H88 receipts")
    weights = version / "adapter" / "adapter_model.safetensors"
    if (weights.is_symlink() or not weights.is_file()
            or sha256_file(weights) != sha(receipt.get("adapter_safetensors_sha256"))):
        raise ValueError("adapter weights digest differs from trained receipt")
    result = read_json(output / "training-result.json")
    config = result.get("configuration")
    expected = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "platform": "kaggle", "max_steps": 1, "precision": "fp16",
                "batch_size": 1, "gradient_accumulation": 1,
                "lora_rank": 8, "audit_records": 32}
    if (not isinstance(config, dict)
            or any(type(config.get(k)) is not type(v) or config.get(k) != v
                   for k, v in expected.items())):
        raise ValueError("wrong exact noncommercial H87 research configuration")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (state.get("config_sha256") != config_sha
            or result.get("schema") != "hala.training.run.v1"
            or result.get("adapter_qualified") is not False
            or result.get("evaluation_required") is not True
            or result.get("checkpoint") != version.name
            or any(result.get(k) != state.get(k) for k in (
                "config_sha256", "dataset_sha256", "manifest_sha256", "runtime",
                "metrics", "changed_parameter_tensors", "peak_vram_bytes"))):
        raise ValueError("training finalization differs from optimizer state")
    runtime = state.get("runtime")
    rows = runtime.get("qualification_steps") if isinstance(runtime, dict) else None
    if (not isinstance(rows, list) or len(rows) != 1
            or not isinstance(rows[0], dict)
            or type(rows[0].get("step")) is not int or rows[0]["step"] != 1
            or not positive(rows[0].get("gradient_norm_before_clip"))
            or type(rows[0].get("changed_parameter_tensors_this_step")) is not int
            or rows[0]["changed_parameter_tensors_this_step"] <= 0
            or not positive(state.get("changed_parameter_tensors"))
            or receipt.get("changed_parameter_tensors") != state["changed_parameter_tensors"]
            or not positive(receipt.get("gradient_norm_before_clip"))
            or receipt["gradient_norm_before_clip"] != rows[0]["gradient_norm_before_clip"]
            or runtime.get("cuda_available") is not True
            or runtime.get("status") != "tested"
            or not isinstance(runtime.get("devices"), list)
            or not runtime["devices"]
            or not all(isinstance(d, dict) and d.get("tensor_gradient_changed") is True
                       for d in runtime["devices"])):
        raise ValueError("no positive first Qwen GPU gradient/parameter update evidence")
    metrics = state.get("metrics")
    if (not isinstance(metrics, list) or len(metrics) != 1
            or not isinstance(metrics[0], dict) or not positive(metrics[0].get("loss"))):
        raise ValueError("first optimizer step has no finite positive model loss")
    return {"schema": SCHEMA, "status": "DOWNLOADED_CHECKPOINT_BYTES_VERIFIED_ONLY",
            "checkpoint_sha256": metadata_sha,
            "adapter_weights_sha256": sha256_file(weights),
            "dataset_sha256": TRAIN_SHA, "optimizer_step": 1,
            "gradient_norm_before_clip": rows[0]["gradient_norm_before_clip"],
            "changed_parameter_tensors": state["changed_parameter_tensors"],
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "provider_private_readback_verified": False,
            "remote_provider_provenance_verified": False,
            "heldout_gui_evaluation_completed": False,
            "commercial_use": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit_downloaded_output(args.downloaded_output), sort_keys=True))


if __name__ == "__main__":
    main()

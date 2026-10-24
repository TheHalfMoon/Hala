"""H90: salvage a complete private Kaggle checkpoint after receipt interruption.

This is CPU-only forensic recovery. It never generates a successful H87/H88
training receipt or claims independent remote provider provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from h73.kaggle_h73_cpu import TRAIN_SHA
from h89.readback import read_json, sha, positive
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

SCHEMA = "hala.h90.interrupted-kaggle-local-checkpoint-candidate.v1"
CHECKPOINT = "hala-h88-research-checkpoint"
CONFIG_SOURCE = Path(__file__).resolve().parents[1] / "experiments/runtime/h61-colab-train-config.json"


def expected_research_configuration():
    original = read_json(CONFIG_SOURCE)
    if (original.get("model_id") != MODEL_ID
            or original.get("revision") != MODEL_REVISION
            or original.get("platform") != "colab"
            or original.get("max_steps") != 600):
        raise ValueError("H61 pinned research input configuration changed")
    config = dict(original)
    config.update(platform="kaggle", max_steps=1,
                  audit_records=32, gradient_accumulation=1)
    return config


def admit_interrupted_checkpoint(downloaded_output):
    root = Path(downloaded_output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("downloaded private Kaggle output directory required")
    output = root / CHECKPOINT
    if output.is_symlink() or not output.is_dir():
        raise ValueError("missing private Kaggle checkpoint directory")
    # H61 original immutable pointer and exact SHA-verified artifact-set proof.
    # The directory may contain a valid completed version but have no final
    # training result or H87/H88 JSON because the notebook was interrupted.
    version, state = load_checkpoint(output, expected=None)
    pointer = read_json(output / "latest.json", budget=1024)
    metadata_sha = sha256_file(version / "checkpoint.json")
    config = expected_research_configuration()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (metadata_sha != sha(pointer.get("checkpoint_sha256"))
            or state.get("config_sha256") != config_hash
            or state.get("dataset_sha256") != TRAIN_SHA
            or type(state.get("step")) is not int or state["step"] != 1
            or type(state.get("max_steps")) is not int or state["max_steps"] != 1
            or type(state.get("changed_parameter_tensors")) is not int
            or state["changed_parameter_tensors"] <= 0):
        raise ValueError("published checkpoint is not the pinned first research step")
    # Confirm the exact original Qwen source and LoRA settings, not just a
    # plausible hash from a different study.
    if (config["model_id"] != MODEL_ID or config["revision"] != MODEL_REVISION
            or config["platform"] != "kaggle"
            or config["batch_size"] != 1 or config["lora_rank"] != 8
            or config["precision"] != "fp16"):
        raise ValueError("not the approved research pilot")
    runtime = state.get("runtime")
    steps = runtime.get("qualification_steps") if isinstance(runtime, dict) else None
    if (not isinstance(runtime, dict)
            or runtime.get("status") != "tested"
            or runtime.get("cuda_available") is not True
            or not isinstance(runtime.get("devices"), list)
            or not runtime["devices"]
            or not isinstance(steps, list) or len(steps) != 1
            or not isinstance(steps[0], dict)
            or type(steps[0].get("step")) is not int or steps[0]["step"] != 1
            or type(steps[0].get("changed_parameter_tensors_this_step")) is not int
            or steps[0]["changed_parameter_tensors_this_step"] <= 0
            or not positive(steps[0].get("gradient_norm_before_clip"))):
        raise ValueError("no complete claimed CUDA gradient-bearing model step")
    measurements = state.get("metrics")
    if (not isinstance(measurements, list) or len(measurements) != 1
            or not isinstance(measurements[0], dict)
            or not positive(measurements[0].get("loss"))):
        raise ValueError("first-step model loss unavailable")
    weights = version / "adapter" / "adapter_model.safetensors"
    optimizer = version / "resume.pt"
    if (weights.is_symlink() or not weights.is_file()
            or optimizer.is_symlink() or not optimizer.is_file()):
        raise ValueError("incomplete adapter/optimizer checkpoint")
    # If finalization exists, it must match exactly; if absent, preserve the
    # validated immutable optimizer version, rather than forging finalization.
    final_path = output / "training-result.json"
    finalized = final_path.exists() or final_path.is_symlink()
    if finalized:
        result = read_json(final_path)
        if (result.get("schema") != "hala.training.run.v1"
                or result.get("checkpoint") != version.name
                or result.get("configuration") != config
                or result.get("adapter_qualified") is not False
                or result.get("evaluation_required") is not True
                or any(result.get(name) != state.get(name) for name in (
                    "config_sha256", "dataset_sha256", "manifest_sha256", "runtime",
                    "metrics", "changed_parameter_tensors", "peak_vram_bytes"))):
            raise ValueError("present run finalization contradicts optimizer checkpoint")
    return {
        "schema": SCHEMA,
        "status": "RECOVERABLE_LOCAL_BYTES_ONLY_NEEDS_PROVIDER_PROOF",
        "checkpoint_sha256": metadata_sha,
        "adapter_sha256": sha256_file(weights),
        "optimizer_sha256": sha256_file(optimizer),
        "dataset_sha256": TRAIN_SHA,
        "research_model_id": MODEL_ID,
        "research_model_revision": MODEL_REVISION,
        "optimizer_step": 1,
        "claimed_positive_gradient_norm": steps[0]["gradient_norm_before_clip"],
        "train_result_present": finalized,
        "provider_authenticated_readback_verified": False,
        "model_training_independently_verified": False,
        "model_quality_evaluated": False,
        "commercial_use": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit_interrupted_checkpoint(args.downloaded_output),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

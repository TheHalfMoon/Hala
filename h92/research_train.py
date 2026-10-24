"""H92 private Kaggle research: 8 NEW Qwen LoRA steps seeded from verified H87.

This is a fresh optimizer run warm-starting H87's verified 1-step adapter.
It is NOT an 8-step continuation of H87 optimizer state, not a trained-model
quality evaluation, and never publishes weights publicly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from h73.kaggle_h73_cpu import ARCHIVE_SHA, H71_CODE_SHA, TRAIN_SHA, locate_attached_train
from h81.staged_resume import admit_stage
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION
from training.scripts.train_adapter import train

KAGGLE_H87_SLUG = "hala-h87-private-original-h71-first-qwen-step"
H87_CHECKPOINT_SHA = "e4f138ba892fe3aa431c4894354479737af3829e99ab47895ff63c6a30112660"
H87_ADAPTER_SHA = "d763a9dafbf4359d75e2411a50ba89e13e1c1009d55f994a13330ca4c3230594"
H92_STEPS = 8
SCHEMA = "hala.h92.private-kaggle-warmstart-eight-new-qwen-steps.v1"


def admit_h87_seed(root: Path) -> tuple[Path, str]:
    """Verify the REAL attached private H87 checkpoint, not an arbitrary adapter."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("private Kaggle inputs root required")
    candidates = []
    for pointer in root.rglob("latest.json"):
        if pointer.is_symlink() or KAGGLE_H87_SLUG not in pointer.as_posix():
            continue
        checkpoint = pointer.parent
        if checkpoint.name != "hala-h87-research-checkpoint":
            continue
        candidates.append(checkpoint)
    if len(candidates) != 1:
        raise ValueError("exactly one attached private H87 checkpoint input is required")
    version, state = load_checkpoint(candidates[0], None)
    adapter = version / "adapter"
    weights = adapter / "adapter_model.safetensors"
    if (not weights.is_file() or weights.is_symlink()
            or not (adapter / "adapter_config.json").is_file()
            or sha256_file(weights) != H87_ADAPTER_SHA
            or sha256_file(version / "checkpoint.json") != H87_CHECKPOINT_SHA
            or state["dataset_sha256"] != TRAIN_SHA
            or state["step"] != 1 or state["max_steps"] != 1
            or state["changed_parameter_tensors"] <= 0):
        raise ValueError("H87 model-initialization bytes not qualified")
    return adapter, H87_ADAPTER_SHA


def research_configuration(repo: Path) -> dict:
    original = json.loads((Path(repo) / "experiments/runtime/h61-colab-train-config.json").read_text("utf8"))
    if (original.get("model_id") != MODEL_ID
            or original.get("revision") != MODEL_REVISION
            or original.get("platform") != "colab"
            or original.get("max_steps") != 600
            or original.get("lora_rank") != 8
            or original.get("precision") != "fp16"):
        raise ValueError("original pinned H61 Qwen experiment drifted")
    return {**original, "platform":"kaggle", "max_steps":H92_STEPS,
            "audit_records":32, "gradient_accumulation":1}


def run(*, repo: Path, scratch: Path, output: Path, attached_inputs: Path = Path("/kaggle/input")) -> dict:
    if (platform.system() != "Linux"
            or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH") != "1"
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get("HALA_REMOTE_TRAINING_ALLOWED") != "1"
            or not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")
            or output.exists()):
        raise ValueError("approved private Kaggle noncommercial fresh optimizer required")
    adapter, adapter_sha = admit_h87_seed(attached_inputs)
    work = scratch / "run"
    admitted = admit_stage(locate_attached_train(), ARCHIVE_SHA, work, H71_CODE_SHA, rederive=True)
    if (admitted.get("status") != "RESTORED_TRAIN_STAGE"
            or admitted.get("dataset_sha256") != TRAIN_SHA):
        raise ValueError("immutable original H71 train source admission failed")
    config = research_configuration(repo)
    cfg_path = scratch / "h92-research-config.json"
    cfg_path.write_text(json.dumps(config,sort_keys=True),encoding="utf8")
    started = time.monotonic()
    train(str(cfg_path),
          str(work / "export/h61_drag_curriculum.jsonl"),
          str(work / "export/h61_drag_curriculum_manifest.json"),
          str(work / "public-dev-baseline-receipt.json"),
          str(output),init_adapter=str(adapter))
    version, state = load_checkpoint(output,None)
    rows = state.get("runtime",{}).get("qualification_steps",[])
    if (state["step"] != H92_STEPS or state["max_steps"] != H92_STEPS
            or len(rows) != H92_STEPS
            or len(state["metrics"]) != H92_STEPS
            or state["changed_parameter_tensors"] <= 0
            or any(type(r.get("step")) is not int or r["step"] != i + 1
                   or type(r.get("changed_parameter_tensors_this_step")) is not int
                   or r["changed_parameter_tensors_this_step"] <= 0
                   or not math.isfinite(r.get("gradient_norm_before_clip",float("nan")))
                   or r["gradient_norm_before_clip"] <= 0
                   for i,r in enumerate(rows))):
        raise ValueError("eight real Qwen optimizer updates not established")
    receipt = {
        "schema":SCHEMA,"status":"EIGHT_NEW_QWEN_LORA_OPTIMIZER_STEPS_VERIFIED",
        "provider":"kaggle_private_personal_noncommercial",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "training_dataset_sha256":TRAIN_SHA,
        "h87_init_checkpoint_sha256":H87_CHECKPOINT_SHA,
        "h87_init_adapter_sha256":adapter_sha,
        "warm_started_from_h87":True,
        "optimizer_state_resumed_from_h87":False,
        "new_optimizer_steps":H92_STEPS,
        "positive_gradient_steps":len(rows),
        "changed_parameter_tensors":state["changed_parameter_tensors"],
        "training_losses":[r["loss"] for r in state["metrics"]],
        "checkpoint_sha256":sha256_file(version / "checkpoint.json"),
        "adapter_sha256":sha256_file(version / "adapter/adapter_model.safetensors"),
        "optimizer_sha256":sha256_file(version / "resume.pt"),
        "checkpoint_version":version.name,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "private_provider_readback_verified":False,
        "heldout_gui_quality_evaluated":False,
        "commercial_use":False
    }
    (Path("/kaggle/working") / "hala-h92-training-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False),encoding="utf8")
    return receipt

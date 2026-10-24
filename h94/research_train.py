"""H94 private/noncommercial Qwen pilot: 64 new optimizer steps from real H92 adapter.

Use unchanged, source-verified original H71 TRAIN, original H61 adapter
trainer and immutable checkpoint publisher; never train on official EVAL.
The optimizer is fresh and no inference/quality success is implied.
"""
import json
import math
import os
import platform
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA, H71_CODE_SHA, TRAIN_SHA, locate_attached_train
from h81.staged_resume import admit_stage
from h92.research_train import research_configuration as h92_research_configuration
from h93.eval import admit_h92_adapter, H92_ADAPTER_SHA, H92_CHECKPOINT_SHA
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION
from training.scripts.train_adapter import train

STEPS = 64
SCHEMA = "hala.h94.private-noncommercial-h92-seed-64-step-qwen.v1"


def research_configuration(repo):
    config = h92_research_configuration(repo)
    if (config["platform"] != "kaggle" or config["max_steps"] != 8
            or config["model_id"] != MODEL_ID or config["revision"] != MODEL_REVISION
            or config["lora_rank"] != 8 or config["precision"] != "fp16"
            or config["gradient_accumulation"] != 1):
        raise ValueError("pinned source configuration no longer matches qualified H92")
    return {**config, "max_steps": STEPS}


def require_research_runtime(*, env=None, operating_system=None,
                             scratch=None, output=None):
    env = os.environ if env is None else env
    operating_system = platform.system() if operating_system is None else operating_system
    if (operating_system != "Linux" or env.get("KAGGLE_KERNEL_RUN_TYPE") is None
            or env.get("HALA_NONCOMMERCIAL_RESEARCH") != "1"
            or env.get("HALA_REMOTE_TRAINING_ALLOWED") != "1"
            or scratch is None or output is None
            or not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")
            or Path(output).exists()):
        raise ValueError("owner-private explicitly noncommercial Kaggle GPU research required")


def run(*, repo, scratch, output, attached_inputs=Path("/kaggle/input")):
    repo, scratch, output = Path(repo), Path(scratch), Path(output)
    require_research_runtime(scratch=scratch, output=output)
    # This reuses H93's SHA-bound, H92 optimizer/state/adapter source admission.
    adapter = admit_h92_adapter(Path(attached_inputs), repo)
    stage = admit_stage(locate_attached_train(), ARCHIVE_SHA,
                        scratch / "run", H71_CODE_SHA, rederive=True)
    if (stage.get("status") != "RESTORED_TRAIN_STAGE"
            or stage.get("dataset_sha256") != TRAIN_SHA):
        raise ValueError("original owner-private H71 source admission failed")
    config = research_configuration(repo)
    config_path = scratch / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), "utf8")
    start = time.monotonic()
    train(str(config_path), str(scratch / "run/export/h61_drag_curriculum.jsonl"),
          str(scratch / "run/export/h61_drag_curriculum_manifest.json"),
          str(scratch / "run/public-dev-baseline-receipt.json"),
          str(output), init_adapter=str(adapter))
    version, state = load_checkpoint(output, None)
    grads = state.get("runtime", {}).get("qualification_steps", [])
    metrics = state.get("metrics", [])
    if (state.get("step") != STEPS or state.get("max_steps") != STEPS
            or len(grads) != STEPS or len(metrics) != STEPS
            or type(state.get("changed_parameter_tensors")) is not int
            or state["changed_parameter_tensors"] <= 0):
        raise ValueError("64 real Qwen optimizer updates not confirmed")
    for i, (g, m) in enumerate(zip(grads, metrics), 1):
        if (type(g.get("step")) is not int or g["step"] != i
                or type(g.get("changed_parameter_tensors_this_step")) is not int
                or g["changed_parameter_tensors_this_step"] <= 0
                or type(g.get("gradient_norm_before_clip")) not in (float, int)
                or not math.isfinite(g["gradient_norm_before_clip"])
                or g["gradient_norm_before_clip"] <= 0
                or type(m.get("loss")) not in (float, int)
                or not math.isfinite(m["loss"]) or m["loss"] <= 0):
            raise ValueError("no finite positive gradient or loss in H94 step")
    receipt = {
        "schema": SCHEMA, "status": "H94_REAL_64_STEP_CHECKPOINT_WRITTEN",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset_sha256": TRAIN_SHA,
        "h92_init_adapter_sha256": H92_ADAPTER_SHA,
        "h92_init_checkpoint_sha256": H92_CHECKPOINT_SHA,
        "optimizer_fresh": True, "steps": STEPS,
        "positive_gradient_steps": len(grads),
        "changed_parameter_tensors": state["changed_parameter_tensors"],
        "first_training_loss": metrics[0]["loss"],
        "last_training_loss": metrics[-1]["loss"],
        "checkpoint_version": version.name,
        "checkpoint_sha256": sha256_file(version / "checkpoint.json"),
        "adapter_sha256": sha256_file(version / "adapter/adapter_model.safetensors"),
        "optimizer_sha256": sha256_file(version / "resume.pt"),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "provider_remote_readback_verified": False,
        "gui_model_quality_evaluated": False,
        "commercial_use": False
    }
    (Path("/kaggle/working") / "hala-h94-training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), "utf8")
    return receipt

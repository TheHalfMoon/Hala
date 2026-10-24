"""H96: owner-private research probe for ONE true vision-target LoRA step."""
from __future__ import annotations
import json
import math
import os
import platform
import time
from pathlib import Path
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h94.research_train import research_configuration as h94_config
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

STEPS=1
SCHEMA="hala.h96.private-vision-lora-real-first-step.v1"

def vision_pilot_config(repo):
    config=h94_config(repo)
    if config.get("max_steps")!=64 or config.get("platform")!="kaggle":
        raise ValueError("H94 source training configuration changed")
    return {**config,"max_steps":1,"audit_records":32}

def require_private_vision_research(*,env=None,operating_system=None,scratch=None,output=None):
    env=os.environ if env is None else env
    operating_system=platform.system() if operating_system is None else operating_system
    if (operating_system!="Linux" or not env.get("KAGGLE_KERNEL_RUN_TYPE")
            or env.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
            or env.get("HALA_REMOTE_TRAINING_ALLOWED")!="1"
            or scratch is None or output is None
            or not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")
            or Path(output).exists()):
        raise ValueError("actual private noncommercial Kaggle vision-only research required")

def run(*,repo,scratch,output):
    repo,scratch,output=Path(repo),Path(scratch),Path(output)
    require_private_vision_research(scratch=scratch,output=output)
    config=vision_pilot_config(repo)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"source",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original H71 TRAIN source failed independent source admission")
    config_path=scratch/"vision-one-step-config.json"
    config_path.write_text(json.dumps(config,sort_keys=True),"utf8")
    start=time.monotonic()
    train(str(config_path),
          str(scratch/"source/export/h61_drag_curriculum.jsonl"),
          str(scratch/"source/export/h61_drag_curriculum_manifest.json"),
          str(scratch/"source/public-dev-baseline-receipt.json"),
          str(output),vision_lora=True)
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    gradients=runtime.get("qualification_steps",[])
    losses=state.get("metrics",[])
    expected={f"model.visual.blocks.{block}.attn.{layer}"
              for block in range(20,24) for layer in ("qkv","proj")}
    selected=runtime.get("h96_vision_target_modules",[])
    if (state.get("step")!=1 or state.get("max_steps")!=1
            or runtime.get("h96_vision_lora_only") is not True
            or set(selected)!=expected or len(selected)!=8
            or runtime.get("trainable_parameter_tensors")!=16
            or len(gradients)!=1 or len(losses)!=1
            or type(gradients[0].get("changed_parameter_tensors_this_step")) is not int
            or gradients[0]["changed_parameter_tensors_this_step"]<=0
            or type(gradients[0].get("gradient_norm_before_clip")) not in (int,float)
            or not math.isfinite(gradients[0]["gradient_norm_before_clip"])
            or gradients[0]["gradient_norm_before_clip"]<=0
            or type(losses[0].get("loss")) not in (int,float)
            or not math.isfinite(losses[0]["loss"]) or losses[0]["loss"]<=0):
        raise ValueError("real gradient-bearing one-step visual LoRA checkpoint not qualified")
    receipt={"schema":SCHEMA,"status":"REAL_ONE_STEP_VISION_LORA_CHECKPOINT_WRITTEN",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "dataset_sha256":TRAIN_SHA,"trainable_scope":"vision_last_four_blocks_only",
        "target_modules":selected,"optimizer_steps":1,
        "changed_parameter_tensors":state["changed_parameter_tensors"],
        "gradient_norm_before_clip":gradients[0]["gradient_norm_before_clip"],
        "train_loss":losses[0]["loss"],"peak_vram_bytes":state.get("peak_vram_bytes"),
        "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
        "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
        "optimizer_sha256":sha256_file(version/"resume.pt"),
        "checkpoint_version":version.name,
        "elapsed_seconds":round(time.monotonic()-start,2),
        "gui_model_quality_evaluated":False,"commercial_use":False,
        "downloaded_host_bytes_verified":False,
        "provider_signed_attestation":False,"production_qualified":False}
    (Path("/kaggle/working")/"hala-h96-vision-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True),"utf8")
    return receipt

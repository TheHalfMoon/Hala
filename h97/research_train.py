"""H97: 32-step vision ONLY LoRA pilot, independent of H94 language adapter.

Uses original real private H71 TRAIN source, never EVAL, and fresh AdamW.
The first-stage H96 success establishes feasibility but H97 trains a NEW
separate vision adapter from pretrained Qwen rather than continuing AdamW.
"""
import json
import math
import os
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import vision_pilot_config,require_private_vision_research
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

STEPS=32
SCHEMA="hala.h97.private-visual-last-four-qwen32.v1"

def training_config(repo):
    c=vision_pilot_config(repo)
    if c.get("max_steps")!=1 or c.get("supervision")!="coordinate_weighted" or c.get("coordinate_weight")!=8.0:
        raise ValueError("original H96 source-qualified weighted-coordinate vision study changed")
    return {**c,"max_steps":STEPS}

def run(*,repo,scratch,output):
    repo,scratch,output=Path(repo),Path(scratch),Path(output)
    require_private_vision_research(scratch=scratch,output=output)
    config=training_config(repo)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"source",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original owner-private H71 TRAIN admission failed")
    cfg=scratch/"visual-32-step-config.json"
    cfg.write_text(json.dumps(config,sort_keys=True),"utf8")
    start=time.monotonic()
    train(str(cfg),str(scratch/"source/export/h61_drag_curriculum.jsonl"),
          str(scratch/"source/export/h61_drag_curriculum_manifest.json"),
          str(scratch/"source/public-dev-baseline-receipt.json"),
          str(output),vision_lora=True)
    version,state=load_checkpoint(output,None)
    r=state.get("runtime",{})
    grads=r.get("qualification_steps",[])
    losses=state.get("metrics",[])
    expected={f"model.visual.blocks.{i}.attn.{module}"
              for i in range(20,24) for module in ("qkv","proj")}
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or r.get("h96_vision_lora_only") is not True
            or set(r.get("h96_vision_target_modules",[]))!=expected
            or r.get("trainable_parameter_tensors")!=16
            or len(grads)!=STEPS or len(losses)!=STEPS
            or type(state.get("changed_parameter_tensors")) is not int
            or state["changed_parameter_tensors"]<=0):
        raise ValueError("H97 32-step vision-only trainable checkpoint not qualified")
    for i,(g,loss) in enumerate(zip(grads,losses),start=1):
        if (type(g.get("step")) is not int or g["step"]!=i
                or type(g.get("changed_parameter_tensors_this_step")) is not int
                or g["changed_parameter_tensors_this_step"]<=0
                or type(g.get("gradient_norm_before_clip")) not in (int,float)
                or not math.isfinite(g["gradient_norm_before_clip"])
                or g["gradient_norm_before_clip"]<=0
                or type(loss.get("loss")) not in (float,int)
                or not math.isfinite(loss["loss"]) or loss["loss"]<=0):
            raise ValueError("missing real vision grad/optimizer update at step "+str(i))
    receipt={"schema":SCHEMA,"status":"REAL_32_VISION_LORA_CHECKPOINT_WRITTEN",
             "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
             "dataset_sha256":TRAIN_SHA,
             "target_modules":sorted(expected),
             "vision_trainable_tensor_count":16,
             "new_optimizer_steps":STEPS,"gradient_bearing_steps":len(grads),
             "changed_parameter_tensors":state["changed_parameter_tensors"],
             "first_training_loss":losses[0]["loss"],
             "final_training_loss":losses[-1]["loss"],
             "peak_vram_bytes":state["peak_vram_bytes"],
             "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
             "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
             "optimizer_sha256":sha256_file(version/"resume.pt"),
             "checkpoint_version":version.name,
             "fresh_base_qwen_no_language_adapter":True,
             "elapsed_seconds":round(time.monotonic()-start,1),
             "gui_quality_evaluated":False,"public_model_released":False,
             "commercial_use":False,
             "independent_host_sha_readback_verified":False}
    (Path("/kaggle/working")/"hala-h97-training-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

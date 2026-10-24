"""H99: source-bound 128-step balanced-language/geometry visual LoRA pilot.

This is a new optimizer from pretrained Qwen: NOT H97 optimizer continuity.
Source, labels and media remain the original immutable H71 TRAIN export.
"""
import hashlib
import json
import math
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h97.research_train import training_config as h97_config
from h96.research_probe import require_private_vision_research
from h99.sampler import STRATEGY,build_balanced_plan
from h61.curriculum import load_export as load_original_h71
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

STEPS=128
SCHEMA="hala.h99.original-h71-balanced-visual-128-step-research.v1"

def training_config(repo):
    config=h97_config(repo)
    if (config.get("max_steps")!=32 or config.get("platform")!="kaggle"
            or config.get("coordinate_weight")!=8.0
            or config.get("supervision")!="coordinate_weighted"
            or config.get("lora_rank")!=8 or config.get("gradient_accumulation")!=1
            or config.get("batch_size")!=1 or config.get("precision")!="fp16"):
        raise ValueError("H97 source-bound visual-only weighted-coordinate contract changed")
    return {**config,"max_steps":STEPS}

def run(*,repo,scratch,output):
    repo,scratch,output=Path(repo),Path(scratch),Path(output)
    require_private_vision_research(scratch=scratch,output=output)
    config=training_config(repo)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"original-H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original immutable H71 TRAIN admission failed")
    training=scratch/"original-H71/export/h61_drag_curriculum.jsonl"
    manifest=scratch/"original-H71/export/h61_drag_curriculum_manifest.json"
    rows=load_original_h71(training,json.loads(manifest.read_text("utf8")))
    plan=build_balanced_plan(rows,STEPS)
    if (len(rows)!=10000 or len(plan.indices)!=STEPS
            or set(plan.summary["selected_languages"])!={"ar","en"}
            or len(plan.summary["selected_families"])<4
            or plan.summary["unique_source_images_selected"]!=STEPS
            or len(plan.summary["selected_geometry"])<6):
        raise ValueError("H99 actual H71 original TRAIN diversity admission failed")
    # This aggregate-only preview is private, before any GPU model allocation.
    preview={"schema":"hala.h99.original-H71-real-prefix-preflight.v1",
             "status":"ACTUAL_SOURCE_DIVERSE_TRAIN_PLAN_VERIFIED",
             "source_original_train_sha256":TRAIN_SHA,
             "selected_plan":plan.summary,
             "commercial_use":False,"model_training_executed":False}
    (Path("/kaggle/working")/"hala-h99-sampler-preflight.json").write_text(
        json.dumps(preview,sort_keys=True,indent=2),"utf8")
    config_path=scratch/"h99-visual-balanced-config.json"
    config_path.write_text(json.dumps(config,sort_keys=True),"utf8")
    start=time.monotonic()
    train(str(config_path),str(training),str(manifest),
          str(scratch/"original-H71/public-dev-baseline-receipt.json"),
          str(output),vision_lora=True,sampling_strategy=STRATEGY)
    version,state=load_checkpoint(output,None)
    r=state.get("runtime",{})
    details=r.get("h99_training_sampler",{})
    steps=r.get("qualification_steps",[])
    losses=state.get("metrics",[])
    selected=[m["record_id"] for step in steps for m in step["microbatches"]]
    expect=[rows[i]["id"] for i in plan.indices]
    target={f"model.visual.blocks.{i}.attn.{n}"
            for i in range(20,24) for n in ("qkv","proj")}
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or r.get("h96_vision_lora_only") is not True
            or set(r.get("h96_vision_target_modules",[]))!=target
            or r.get("trainable_parameter_tensors")!=16
            or len(steps)!=STEPS or len(losses)!=STEPS
            or selected!=expect or len(set(selected))!=STEPS
            or details.get("plan_sha256")!=plan.sha256
            or details.get("selected_languages")!=plan.summary["selected_languages"]
            or details.get("selected_families")!=plan.summary["selected_families"]
            or state.get("dataset_sha256")!=TRAIN_SHA):
        raise ValueError("H99 actual optimizer samples, visual model or identity not proven")
    for i,(step,loss) in enumerate(zip(steps,losses),1):
        g=step.get("gradient_norm_before_clip")
        if (step.get("step")!=i or step.get("changed_parameter_tensors_this_step",0)<=0
                or type(g) not in (float,int) or not math.isfinite(g) or g<=0
                or loss.get("step")!=i or not math.isfinite(loss.get("loss",float("nan")))
                or loss["loss"]<=0):
            raise ValueError("unverified real H99 gradient update "+str(i))
    receipt={"schema":SCHEMA,"status":"REAL_BALANCED_128_VISUAL_UPDATES_RECORDED",
             "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
             "original_dataset_sha256":TRAIN_SHA,
             "sampler_plan_sha256":plan.sha256,
             "source_records_total":len(rows),
             "actual_unique_optimizer_records":len(set(selected)),
             "sampling_summary":plan.summary,
             "new_optimizer_steps":STEPS,
             "positive_gradient_steps":len(steps),
             "trainable_visual_tensors":r["trainable_parameter_tensors"],
             "changed_visual_tensors":state["changed_parameter_tensors"],
             "first_training_loss":losses[0]["loss"],
             "final_training_loss":losses[-1]["loss"],
             "peak_vram_bytes":state["peak_vram_bytes"],
             "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
             "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
             "optimizer_sha256":sha256_file(version/"resume.pt"),
             "checkpoint_version":version.name,
             "elapsed_seconds":round(time.monotonic()-start,2),
             "fresh_base_qwen_not_h97_contiguous_optimizer":True,
             "independent_GUI_quality_verified":False,
             "private_download_SHA_verified":False,
             "commercial_use":False,"public_model_released":False}
    (Path("/kaggle/working")/"hala-h99-training-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

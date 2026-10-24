"""H106: fresh 256-step original H105 TRAIN joint vision+language LoRA PILOT.

No merge/publication. First compare strict GUI quality to fresh identical base
and H99 after independently downloading private original model attempts.
"""
import json
import math
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h99.research_train import training_config as h99_config
from h61.curriculum import load_export as load_h71
from h106.training_plan import STEPS,STRATEGY,get_plan,H105_QUALIFIED_PLAN_SHA
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h106.owner-private-256-joint-source-grounding-pilot.v1"


def config(repo):
    cfg=h99_config(repo)
    if (cfg["max_steps"]!=128 or cfg["coordinate_weight"]!=8.0
            or cfg["supervision"]!="coordinate_weighted"
            or cfg["lora_rank"]!=8 or cfg["precision"]!="fp16"
            or cfg["batch_size"]!=1 or cfg["gradient_accumulation"]!=1):
        raise ValueError("original H99 source train contract changed")
    return {**cfg,"max_steps":STEPS}


def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    cfg=config(repo)
    admitted=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                         scratch/"original-H71",H71_CODE_SHA,rederive=True)
    if admitted["status"]!="RESTORED_TRAIN_STAGE" or admitted["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("independent original H71 TRAIN source archive rederivation failed")
    root=scratch/"original-H71/export"
    trainpath=root/"h61_drag_curriculum.jsonl"
    manifestpath=root/"h61_drag_curriculum_manifest.json"
    rows=load_h71(trainpath,json.loads(manifestpath.read_text("utf8")))
    plan=get_plan(rows,root,STEPS)
    if (len(rows)!=10000 or len(plan.indices)!=STEPS or
            plan.summary["parent_original_h105_512_plan_sha256"]!=H105_QUALIFIED_PLAN_SHA):
        raise ValueError("H106 exact H105 512 source plan prefix did not qualify")
    preflight={"schema":"hala.h106.original-h71-private-preflight.v1",
               "selected_original_source_sha256":TRAIN_SHA,
               "256_prefix_h105_provenance":plan.summary,
               "model_training_executed":False,"private_data":True}
    (Path("/kaggle/working")/"hala-h106-source-preflight.json").write_text(
        json.dumps(preflight,sort_keys=True,indent=2),"utf8")
    configpath=scratch/"h106-original-joint-config.json"
    configpath.write_text(json.dumps(cfg,sort_keys=True),"utf8")
    started=time.monotonic()
    train(str(configpath),str(trainpath),str(manifestpath),
          str(scratch/"original-H71/public-dev-baseline-receipt.json"),
          str(output),joint_lora=True,sampling_strategy=STRATEGY)
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    sample=runtime.get("h106_joint_training_sampler",{})
    updates=runtime.get("qualification_steps",[])
    losses=state.get("metrics",[])
    original_ids=[rows[i]["id"] for i in plan.indices]
    actual_ids=[m["record_id"] for step in updates for m in step["microbatches"]]
    visual=runtime.get("h106_trainable_visual_modules",[])
    language=runtime.get("h106_trainable_language_modules",[])
    if (state["step"]!=STEPS or state["max_steps"]!=STEPS
            or sample.get("plan_sha256")!=plan.sha256
            or sample.get("source_split")!="TRAIN_ONLY"
            or actual_ids!=original_ids or len(set(actual_ids))!=STEPS
            or runtime.get("h106_joint_lora_verified") is not True
            or runtime.get("h106_vision_and_language_unfrozen") is not True
            or len(visual)!=8 or not 16<=len(language)<=256
            or runtime.get("trainable_parameter_tensors")!=2*(len(visual)+len(language))
            or len(updates)!=STEPS or len(losses)!=STEPS
            or state["dataset_sha256"]!=TRAIN_SHA
            or state["changed_parameter_tensors"]<=0):
        raise ValueError("real joint optimizer/label/media identity not proven")
    for i,(update,loss) in enumerate(zip(updates,losses),start=1):
        gradient=update.get("gradient_norm_before_clip")
        if (update["step"]!=i or update["changed_parameter_tensors_this_step"]<=0
                or type(gradient) not in (int,float) or not math.isfinite(gradient)
                or gradient<=0 or loss["step"]!=i
                or not math.isfinite(loss["loss"]) or loss["loss"]<=0):
            raise ValueError("H106 train step missing valid gradient/parameter update")
    receipt={
        "schema":SCHEMA,"status":"ORIGINAL_H71_REAL_256_JOINT_OPTIMIZER_STEPS_VERIFIED",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "original_h71_train_sha256":TRAIN_SHA,
        "new_h106_sampler_prefix_sha256":plan.sha256,
        "h105_qualified_parent_plan_sha256":H105_QUALIFIED_PLAN_SHA,
        "new_optimizer_steps":STEPS,
        "verified_unique_original_train_screenshots":len(set(actual_ids)),
        "gradient_bearing_steps":len(updates),
        "trainable_visual_modules":len(visual),
        "trainable_language_modules":len(language),
        "trainable_adapter_tensors":runtime["trainable_parameter_tensors"],
        "changed_adapter_tensors":state["changed_parameter_tensors"],
        "first_training_loss":losses[0]["loss"],
        "final_training_loss":losses[-1]["loss"],
        "peak_vram_bytes":state["peak_vram_bytes"],
        "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
        "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
        "optimizer_sha256":sha256_file(version/"resume.pt"),
        "checkpoint_version":version.name,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "independent_GUI_quality_verified":False,
        "model_improvement_claimed":False,
        "public_checkpoint_released":False,
        "paid_resources_used":False,
        "commercial_use":False}
    (Path("/kaggle/working")/"hala-h106-private-joint-training-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

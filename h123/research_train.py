"""H123 free T4 true 896-step fresh English complex GUI END dual-grid SFT.

Only SHA-qualified, previously unused original H71 TRAIN source images.
No heldout/EVAL/TEST labels in optimizer, no inherited donor adapter.
"""
import json
import math
import time
from pathlib import Path

from h106.research_train import config as inherited_config
from h106.training_plan import STRATEGY as H106_SAMPLER
from h61.curriculum import load_export as load_h71
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h96.research_probe import require_private_vision_research
from h123.research_cpu import run as source_qualify
from h123.schedule import derive,TRAIN,HOLDOUT
from h123.training_plan import (STEPS,TRAIN_SHA,HELDOUT_SHA,
                                SCHEDULE_SHA,get_plan)
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h123.original-complex-TRAIN896-fresh-dual-grid-T4.v1"


def config(repo):
    baseline=inherited_config(Path(repo))
    if (baseline["max_steps"]!=256 or baseline["precision"]!="fp16"
            or baseline["batch_size"]!=1
            or baseline["gradient_accumulation"]!=1
            or baseline["lora_rank"]!=8
            or baseline["supervision"]!="coordinate_weighted"
            or baseline["coordinate_weight"]!=8.0
            or baseline["platform"]!="kaggle"):
        raise ValueError("H123 original free T4 rank8 visual+text config drift")
    return {**baseline,"max_steps":STEPS}


def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    source_receipt=Path("/kaggle/working/hala-h123-train-preflight.json")
    if source_receipt.exists() or output.exists():
        raise ValueError("H123 private nonoverwrite original source train")
    original_source=source_qualify(scratch,source_receipt)
    if (original_source["status"]!=
            "PASS_OWNER_H123_NEW_ORIGINAL_1024_COMPLEX_TRAIN_SOURCE"
            or original_source["train_private_sha256"]!=TRAIN_SHA
            or original_source["heldout_private_sha256"]!=HELDOUT_SHA
            or original_source["schedule_sha256"]!=SCHEDULE_SHA
            or original_source["image_bytes_and_original_END_box_SHA_verified"]!=1024):
        raise ValueError("H123 original 1024 TRAIN image/box source not qualified")
    root=scratch/"H71/export"
    examples=root/"h61_drag_curriculum.jsonl"
    manifest=root/"h61_drag_curriculum_manifest.json"
    original=load_h71(examples,json.loads(manifest.read_text("utf8")))
    selected=derive(original,root)
    plan=get_plan(selected,STEPS)
    if (len(plan.indices)!=TRAIN or len(selected.heldout)!=HOLDOUT
            or plan.summary.get("no_heldout_optimizer_input") is not True):
        raise ValueError("H123 exact new 896/128 source overlap or digest drift")
    params=config(repo)
    paramsfile=scratch/"h123-original-complex896.json"
    paramsfile.write_text(json.dumps(params,sort_keys=True),"utf8")
    started=time.monotonic()
    train(str(paramsfile),str(examples),str(manifest),
          str(scratch/"H71/public-dev-baseline-receipt.json"),
          str(output),sampling_strategy=H106_SAMPLER,joint_lora=True,
          complex_grid_mode=True)
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    updates=runtime.get("qualification_steps",[])
    metrics=state.get("metrics",[])
    observed=[a.get("record_id") for u in updates
              for a in u.get("microbatches",[])]
    expected=[a["id"] for a in selected.train]
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or len(updates)!=STEPS or len(metrics)!=STEPS
            or observed!=expected or len(set(observed))!=STEPS
            or runtime.get("h123_new_complex896_sampler",{}).get(
                "train_private_sha256")!=TRAIN_SHA
            or runtime.get("h123_new_complex896_sampler",{}).get(
                "heldout_private_sha256")!=HELDOUT_SHA
            or runtime.get("h123_new_complex896_sampler",{}).get(
                "schedule_sha256")!=SCHEDULE_SHA
            or runtime.get("h123_original_complex_processor_audits")!=12
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or len(runtime.get("h106_trainable_language_modules",[]))!=56
            or state.get("dataset_sha256")!=H71_SHA
            or state.get("changed_parameter_tensors",0)<=0):
        raise ValueError("H123 actual real 896-step/adapter/source proof incomplete")
    for idx,(u,m) in enumerate(zip(updates,metrics),1):
        norm=u.get("gradient_norm_before_clip")
        if (u.get("step")!=idx or m.get("step")!=idx
                or u.get("changed_parameter_tensors_this_step",0)<=0
                or type(norm) not in (int,float)
                or not math.isfinite(norm) or norm<=0
                or type(m.get("loss")) not in (int,float)
                or not math.isfinite(m["loss"]) or m["loss"]<=0):
            raise ValueError("H123 actual nonzero optimizer gradients required")
    receipt={
        "schema":SCHEMA,
        "status":"H123_REAL896_NEW_COMPLEX_TRAIN_ONLY_OPTIMIZER_VERIFIED",
        "original_H71_TRAIN_SHA":H71_SHA,
        "train_private_sha256":TRAIN_SHA,
        "heldout_private_sha256":HELDOUT_SHA,
        "schedule_sha256":SCHEDULE_SHA,
        "original_source_media_SHA_verified":1024,
        "actual_train_optimizer_steps":STEPS,
        "positive_gradient_steps":STEPS,
        "fresh_original_train_images":TRAIN,
        "never_optimized_heldout_train_images":HOLDOUT,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "final_adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
        "final_checkpoint_sha256":sha256_file(version/"checkpoint.json"),
        "final_optimizer_sha256":sha256_file(version/"resume.pt"),
        "final_checkpoint_version":version.name,
        "initial_loss":metrics[0]["loss"],"last_loss":metrics[-1]["loss"],
        "peak_vram_bytes":state.get("peak_vram_bytes"),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "original_EVAL_or_TEST_in_training":False,
        "previous_H113_H122_source_reused":False,
        "heldout_scored_or_used_for_optimizer":False,
        "model_GUI_skill_qualified":False,
        "model_weights_public":False,
        "paid_compute_used":False,"commercial_use":False}
    (Path("/kaggle/working")/"hala-h123-private-training-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

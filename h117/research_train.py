"""H117 private free-T4 START-conditioned signed destination pilot.

No original model/weights/data modified. No publication or EVAL/TEST in
training. A full owner SHA/optimizer readback and NEW strict GUI evaluation
are mandatory before any model improvement claim.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h106.research_train import config as h106_config
from h61.curriculum import load_export as load_h71
from h114.schedule import (qualify_owner_private,SCREENSHOTS,STEPS,
                           ORIGINAL_H113_DERIVED_SHA,ORIGINAL_H105_TRAIN_PLAN_SHA)
from h114.schedule import trainer_row
from h116.delta import derive_schedule
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h117.owner-private-real-256-conditioned-signed-delta-updates.v1"


def attached_private_points(attached=Path("/kaggle/input")):
    sources=[p for p in Path(attached).rglob("private-source-bound-dual-point-sft.jsonl")
             if ("hala-h113-private-original-dual-point-cpu" in p.as_posix()
                 and not p.is_symlink() and p.is_file())]
    if len(sources)!=1:
        raise ValueError("exactly one owner-private finished H113 source SFT kernel attachment")
    pointfile=sources[0]
    plan=qualify_owner_private(pointfile)
    if (plan.summary["original_h113_point_jsonl_sha256"]!=ORIGINAL_H113_DERIVED_SHA
            or plan.summary["pilot_selected_original_screenshots"]!=128):
        raise ValueError("original H113 authentic TRAIN point data changed")
    return pointfile,plan


def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    points,parent=attached_private_points()
    selected=derive_schedule(parent)
    cfg=h106_config(repo)
    if (cfg["max_steps"]!=STEPS or cfg["lora_rank"]!=8
            or cfg["supervision"]!="coordinate_weighted"
            or cfg["coordinate_weight"]!=8.0
            or cfg["precision"]!="fp16"
            or cfg["batch_size"]!=1 or cfg["gradient_accumulation"]!=1):
        raise ValueError("H117 original 256-step independent point training profile drift")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"original-H71",H71_CODE_SHA,rederive=True)
    if stage["status"]!="RESTORED_TRAIN_STAGE" or stage["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("H117 exact original source H71 TRAIN archive not verified")
    root=scratch/"original-H71/export"
    rows_file=root/"h61_drag_curriculum.jsonl"
    manifest_file=root/"h61_drag_curriculum_manifest.json"
    original=load_h71(rows_file,json.loads(manifest_file.read_text("utf8")))
    if len(original)!=10000:raise ValueError("original H71 immutable source 10k missing")
    from h106.training_plan import get_plan,H105_QUALIFIED_PLAN_SHA
    orig_plan=get_plan(original,root,STEPS)
    if (orig_plan.summary["source_split"]!="TRAIN_ONLY"
            or orig_plan.summary["parent_original_h105_512_plan_sha256"]!=ORIGINAL_H105_TRAIN_PLAN_SHA
            or ORIGINAL_H105_TRAIN_PLAN_SHA!=H105_QUALIFIED_PLAN_SHA):
        raise ValueError("actual original H105 source plan no longer pinned")
    if selected.summary["derived_private_training_jsonl_sha256"]!="174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c":
        raise ValueError("H117 exact original TRAIN conditioned signed destination source drift")
    for i in range(SCREENSHOTS):
        source=original[orig_plan.indices[i]]
        for r in range(2):
            _=trainer_row(source,selected.selected[i*2+r])
    config_path=scratch/"h117-original-train-conditioned-delta-config.json"
    config_path.write_text(json.dumps(cfg,sort_keys=True),"utf8")
    started=time.monotonic()
    train(str(config_path),str(rows_file),str(manifest_file),
          str(scratch/"original-H71/public-dev-baseline-receipt.json"),
          str(output),joint_lora=True,
          sampling_strategy="hala.h106.joint-vision-language-h105-source-prefix256.v1",
          conditioned_delta_sft_path=str(points))
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    steps=runtime.get("qualification_steps",[])
    point_runtime=runtime.get("h117_start_conditioned_delta_sampler",{})
    losses=state.get("metrics",[])
    actual=[x["record_id"] for s in steps for x in s["microbatches"]]
    expected=[x["id"] for x in selected.selected]
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or len(steps)!=STEPS or len(losses)!=STEPS
            or actual!=expected
            or state.get("dataset_sha256")!=TRAIN_SHA
            or point_runtime.get("derived_private_training_jsonl_sha256")!=
                  selected.summary["derived_private_training_jsonl_sha256"]
            or point_runtime.get("original_h113_parent_sha256")!=ORIGINAL_H113_DERIVED_SHA
            or runtime.get("h117_signed_delta_processor_audited_roles")!=10
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or not 16<=len(runtime.get("h106_trainable_language_modules",[]))<=256
            or state.get("changed_parameter_tensors",0)<=0):
        raise ValueError("H117 new original 256-step independent point optimizer not verified")
    for i,(update,loss) in enumerate(zip(steps,losses),1):
        g=update.get("gradient_norm_before_clip")
        if (update.get("step")!=i
                or update.get("changed_parameter_tensors_this_step",0)<=0
                or type(g) not in (int,float) or not math.isfinite(g) or g<=0
                or loss.get("step")!=i
                or not math.isfinite(loss["loss"]) or loss["loss"]<=0):
            raise ValueError("H117 actual optimizer gradient/step source not qualified")
    receipt={
       "schema":SCHEMA,"status":"H117_REAL_256_ORIGINAL_START_CONDITIONED_DELTA_UPDATES_VERIFIED",
       "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
       "original_train_source_sha256":TRAIN_SHA,
       "original_h113_private_dual_point_sha256":ORIGINAL_H113_DERIVED_SHA,
       "original_h116_conditioned_signed_delta_sha256":selected.summary["derived_private_training_jsonl_sha256"],
       "original_h105_parent_source_512_sha256":ORIGINAL_H105_TRAIN_PLAN_SHA,
       "pilot_source_images":selected.summary["source_images"],
       "start_point_training_rows":selected.summary["start_point_roles"],
       "conditioned_delta_training_rows":selected.summary["conditioned_signed_destination_roles"],
       "original_train_languages":selected.summary["original_languages"],
       "actual_optimizer_steps":STEPS,
       "positive_gradient_and_changed_weights_steps":len(steps),
       "joint_visual_layers":len(runtime["h106_trainable_visual_modules"]),
       "joint_language_layers":len(runtime["h106_trainable_language_modules"]),
       "changed_lora_parameter_tensors":state["changed_parameter_tensors"],
       "first_training_loss":losses[0]["loss"],
       "last_training_loss":losses[-1]["loss"],
       "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
       "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
       "optimizer_sha256":sha256_file(version/"resume.pt"),
       "checkpoint_version":version.name,
       "peak_vram_bytes":state["peak_vram_bytes"],
       "elapsed_seconds":round(time.monotonic()-started,2),
       "original_dataset_or_old_weights_modified":False,
       "no_eval_or_test_in_train":True,
       "new_strict_gui_quality_qualified":False,
       "paid_resources_used":False,"commercial_use":False,
       "model_weights_published":False}
    (Path("/kaggle/working")/"hala-h117-private-conditioned-delta-training-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

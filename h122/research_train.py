"""H122 original TRAIN-only 448 true END-grid LoRA updates, Kaggle free T4.

Do NOT use H122 heldout original TRAIN images as optimizer inputs,
nor any prior original DragOn public EVAL samples or outcomes.
"""
import json
import math
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h106.research_train import config as base_config
from h106.training_plan import STRATEGY as H106_STRATEGY
from h61.curriculum import load_export as load_h71
from h122.schedule import qualify_owner_private,TRAIN,HOLDOUT
from h122.training_plan import get_plan,STEPS
from h122.pre_gpu import qualify_before_gpu
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h122.private-real448-original-train-end-grid10-absolute.v1"
TRAIN_SHA_H122="8f5e51a1fd0e06f2c22a30715cb29d5785296733b7401fd3c6807e33c1f376ac"
HELDOUT_SHA_H122="53e85c5e9070b8c1c2f6cce91b30a5e5c6d46505551a30b0090aa52cb90cd074"
SCHEDULE_SHA_H122="f986eeb5ca0c242a1d000ebe4474dc429abdc54514b20dd9e82e1537cbd57ba2"

def config(repo):
    base=base_config(Path(repo))
    if (base["max_steps"]!=256 or base["precision"]!="fp16"
            or base["batch_size"]!=1 or base["gradient_accumulation"]!=1
            or base["lora_rank"]!=8 or base["coordinate_weight"]!=8.0
            or base["supervision"]!="coordinate_weighted"
            or base["platform"]!="kaggle"):
        raise ValueError("H122 inherited free T4 runtime source config drift")
    return {**base,"max_steps":STEPS}

def owner_private_schedule(attached=Path("/kaggle/input")):
    files=[p for p in Path(attached).rglob("private-source-bound-dual-point-sft.jsonl")
           if "hala-h113-private-original-dual-point-cpu" in p.as_posix()
           and p.is_file() and not p.is_symlink()]
    if len(files)!=1:
        raise ValueError("one original H113 owner-private TRAIN source required")
    result=qualify_owner_private(files[0])
    data=result.summary
    if (data.get("train_private_sha256")!=TRAIN_SHA_H122
            or data.get("heldout_private_sha256")!=HELDOUT_SHA_H122
            or data.get("schedule_sha256")!=SCHEDULE_SHA_H122
            or data.get("no_holdout_optimizer_input") is not True):
        raise ValueError("H122 original TRAIN-only 448 /64 frozen split SHA changed")
    return files[0],result

def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    points,qualified=owner_private_schedule()
    cfg=config(repo)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"original-H71",H71_CODE_SHA,rederive=True)
    if stage["status"]!="RESTORED_TRAIN_STAGE" or stage["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("H122 original H71 TRAIN source not qualified")
    root=scratch/"original-H71/export"
    rowsfile=root/"h61_drag_curriculum.jsonl"
    manifest=root/"h61_drag_curriculum_manifest.json"
    original=load_h71(rowsfile,json.loads(manifest.read_text("utf8")))
    plan=get_plan(original,root,qualified,STEPS)
    if len(plan.indices)!=TRAIN or len(qualified.heldout)!=HOLDOUT:
        raise ValueError("H122 original source heldout disjointness missing")
    original_source_proof=qualify_before_gpu(original,root,qualified,plan)
    if (original_source_proof["original_source_images_sha_verified"]!=512
            or original_source_proof["optimizer_train_images_sha_verified"]!=TRAIN
            or original_source_proof["excluded_heldout_train_images_sha_verified"]!=HOLDOUT):
        raise ValueError("H122 all original TRAIN screenshots unverified")
    (Path("/kaggle/working")/"hala-h122-preflight.json").write_text(
        json.dumps({"schema":SCHEMA,"status":"448_TRAIN_64_TRAIN_HELDOUT_SHA_QUALIFIED",
                    "source_h71_train_sha256":TRAIN_SHA,
                    "h122_train_data_sha256":TRAIN_SHA_H122,
                    "h122_heldout_data_sha256":HELDOUT_SHA_H122,
                    "original_source_pre_gpu_proof":original_source_proof,
                    "optimizers_executed":False,
                    "model_improvement_claimed":False,
                    "no_EVAL_or_TEST_in_training":True},
                   sort_keys=True,indent=2),"utf8")
    cfgpath=scratch/"h122-original-train448-config.json"
    cfgpath.write_text(json.dumps(cfg,sort_keys=True),"utf8")
    began=time.monotonic()
    train(str(cfgpath),str(rowsfile),str(manifest),
          str(scratch/"original-H71/public-dev-baseline-receipt.json"),
          str(output),joint_lora=True,sampling_strategy=H106_STRATEGY,
          end_grid_sft_path=str(points))
    version,state=load_checkpoint(output,None)
    runtime=state["runtime"]
    steps=runtime.get("qualification_steps",[])
    metrics=state.get("metrics",[])
    source_ids=[r["record_id"] for step in steps for r in step["microbatches"]]
    expected=[r["id"] for r in qualified.train]
    rejected={r["id"] for r in qualified.heldout}
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or len(steps)!=STEPS or len(metrics)!=STEPS
            or source_ids!=expected or len(set(source_ids))!=STEPS
            or set(source_ids)&rejected
            or runtime.get("h122_end_grid_train_sampler",{}).get(
                "train_private_sha256")!=TRAIN_SHA_H122
            or runtime.get("h122_end_grid_train_sampler",{}).get(
                "heldout_private_sha256")!=HELDOUT_SHA_H122
            or runtime.get("h122_end_processor_audits")!=8
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or len(runtime.get("h106_trainable_language_modules",[]))!=56
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("changed_parameter_tensors",0)<=0):
        raise ValueError("H122 actual 448-step original TRAIN-only END-grid evidence incomplete")
    for index,(step,loss) in enumerate(zip(steps,metrics),1):
        g=step.get("gradient_norm_before_clip")
        if (step.get("step")!=index or loss.get("step")!=index
                or step.get("changed_parameter_tensors_this_step",0)<=0
                or type(g) not in (int,float) or not math.isfinite(g) or g<=0
                or not math.isfinite(loss["loss"]) or loss["loss"]<=0):
            raise ValueError("H122 optimizer gradient/source evidence incomplete")
    result={
        "schema":SCHEMA,"status":"H122_REAL448_TRAIN_ONLY_END_GRID_OPTIMIZER_VERIFIED",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "source_original_h71_SHA":TRAIN_SHA,
        "train_private_sha256":TRAIN_SHA_H122,
        "heldout_private_sha256":HELDOUT_SHA_H122,
        "schedule_sha256":SCHEDULE_SHA_H122,
        "original_source_pre_gpu_proof":original_source_proof,
        "original_train_screenshots":TRAIN,
        "original_heldout_train_screenshots":HOLDOUT,
        "actual_optimizer_steps":STEPS,
        "positive_gradient_steps":STEPS,
        "final_adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
        "final_checkpoint_sha256":sha256_file(version/"checkpoint.json"),
        "final_optimizer_sha256":sha256_file(version/"resume.pt"),
        "final_checkpoint_version":version.name,
        "initial_train_loss":metrics[0]["loss"],
        "last_train_loss":metrics[-1]["loss"],
        "peak_vram_bytes":state["peak_vram_bytes"],
        "elapsed_seconds":round(time.monotonic()-began,2),
        "no_EVAL_or_TEST_used":True,
        "actual_validation_scored":False,
        "model_GUI_skill_qualified":False,
        "model_weights_published":False,"paid_compute":False,
        "commercial_use":False}
    (Path("/kaggle/working")/"hala-h122-private-train-receipt.json").write_text(
        json.dumps(result,sort_keys=True,indent=2),"utf8")
    return result

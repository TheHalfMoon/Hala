"""H119 free private T4 1024-step source-qualified fresh joint LoRA attempt."""
import json
import math
import time
from pathlib import Path

from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h96.research_probe import require_private_vision_research
from h106.research_train import config as h106_config
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h61.curriculum import load_export as load_h71
from h119.curriculum import qualify_owner_private,SCHEMA as H119_SCHEMA
from h119.training_plan import get_plan,STEPS,PAIRS
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION
from h106.training_plan import STRATEGY as H106_STRATEGY

SCHEMA="hala.h119.private-real1024-robust-original-512-train.v1"
EXPECTED_TRAIN_SHA="81fa28ca2f6d20e26b108e3417e1943cc5b8d278582f3c220e456c1fe4ac5ca6"

def config(repo):
    base=h106_config(repo)
    if (base["max_steps"]!=256 or base["precision"]!="fp16"
            or base["batch_size"]!=1 or base["gradient_accumulation"]!=1
            or base["lora_rank"]!=8 or base["coordinate_weight"]!=8.0
            or base["supervision"]!="coordinate_weighted"
            or base["platform"]!="kaggle"):
        raise ValueError("H119 bounded inherited free T4 source config drift")
    return {**base,"max_steps":STEPS}

def attached_private_points(attached=Path("/kaggle/input")):
    paths=[p for p in Path(attached).rglob("private-source-bound-dual-point-sft.jsonl")
           if "hala-h113-private-original-dual-point-cpu" in p.as_posix()
           and p.is_file() and not p.is_symlink()]
    if len(paths)!=1:
        raise ValueError("single unchanged owner-private H113 source required")
    data=qualify_owner_private(paths[0])
    if (data.summary["derived_private_train_sha256"]!=EXPECTED_TRAIN_SHA
            or data.summary["source_split"]!="TRAIN_ONLY"):
        raise ValueError("H119 original TRAIN-only robust source SHA mismatch")
    return paths[0],data

def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    pointfile,selected=attached_private_points()
    cfg=config(repo)
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"original-H71",H71_CODE_SHA,rederive=True)
    if stage["status"]!="RESTORED_TRAIN_STAGE" or stage["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("H119 exact original H71 TRAIN source missing")
    root=scratch/"original-H71/export"
    rowsfile=root/"h61_drag_curriculum.jsonl"
    manifest=root/"h61_drag_curriculum_manifest.json"
    original=load_h71(rowsfile,json.loads(manifest.read_text("utf8")))
    plan=get_plan(original,root,STEPS)
    if (len(original)!=10000 or len(plan.indices)!=PAIRS
            or plan.summary["parent_original_h105_sha256"]!=H105_QUALIFIED_PLAN_SHA):
        raise ValueError("H119 original H105 512 selection mismatch")
    (Path("/kaggle/working")/"hala-h119-source-only-preflight.json").write_text(
        json.dumps({"schema":"hala.h119.source-only-preflight.v1",
                    "status":"ORIGINAL_TRAIN_ONLY_FULL512_QUALIFIED",
                    "source_derived_sha256":EXPECTED_TRAIN_SHA,
                    "original_h71_sha256":TRAIN_SHA,
                    "h105_parent_source_sha256":H105_QUALIFIED_PLAN_SHA,
                    "model_training_executed":False,"eval_or_test_training":False},
                   sort_keys=True,indent=2),"utf8")
    cfgpath=scratch/"h119-private-original1024-config.json"
    cfgpath.write_text(json.dumps(cfg,sort_keys=True),"utf8")
    began=time.monotonic()
    train(str(cfgpath),str(rowsfile),str(manifest),
          str(scratch/"original-H71/public-dev-baseline-receipt.json"),
          str(output),joint_lora=True,sampling_strategy=H106_STRATEGY,
          robust_anchor_sft_path=str(pointfile))
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    updates=runtime.get("qualification_steps",[])
    metrics=state.get("metrics",[])
    actual=[r["record_id"] for s in updates for r in s["microbatches"]]
    expected=[r["id"] for r in selected.selected]
    if (state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or len(updates)!=STEPS or len(metrics)!=STEPS
            or actual!=expected or len(set(actual))!=STEPS
            or runtime.get("h119_full512_robust_anchor_sampler",{}).get(
                "derived_private_train_sha256")!=EXPECTED_TRAIN_SHA
            or runtime.get("h119_full512_original_plan",{}).get(
                "parent_original_h105_sha256")!=H105_QUALIFIED_PLAN_SHA
            or runtime.get("h119_original_role_processor_audits")!=12
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("changed_parameter_tensors",0)<=0):
        raise ValueError("H119 original 1024-step exact optimizer evidence incomplete")
    for i,(step,loss) in enumerate(zip(updates,metrics),1):
        g=step.get("gradient_norm_before_clip")
        if (step["step"]!=i or loss["step"]!=i or
                step["changed_parameter_tensors_this_step"]<=0
                or type(g) not in (int,float) or not math.isfinite(g) or g<=0
                or not math.isfinite(loss["loss"]) or loss["loss"]<=0):
            raise ValueError("H119 actual GPU optimizer step not qualified")
    receipt={
      "schema":SCHEMA,"status":"H119_REAL_1024_ORIGINAL_FULL512_ROBUST_TRAIN_STEPS_VERIFIED",
      "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
      "source_train_sha256":TRAIN_SHA,
      "original_h119_derived_sha256":EXPECTED_TRAIN_SHA,
      "original_h105_512_plan_sha256":H105_QUALIFIED_PLAN_SHA,
      "original_source_images":PAIRS,"train_start_roles":PAIRS,"train_end_delta_roles":PAIRS,
      "actual_optimizer_steps":len(updates),
      "actual_positive_gradient_steps":len(updates),
      "joint_visual_lora_modules":len(runtime["h106_trainable_visual_modules"]),
      "joint_language_lora_modules":len(runtime["h106_trainable_language_modules"]),
      "first_train_loss":metrics[0]["loss"],"last_train_loss":metrics[-1]["loss"],
      "peak_vram_bytes":state["peak_vram_bytes"],
      "adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
      "checkpoint_sha256":sha256_file(version/"checkpoint.json"),
      "optimizer_sha256":sha256_file(version/"resume.pt"),
      "checkpoint_version":version.name,
      "elapsed_seconds":round(time.monotonic()-began,2),
      "no_EVAL_or_TEST_training":True,"model_weights_published":False,
      "new_strict_GUI_improvement_claimed":False,"paid_resources_used":False,
      "commercial_use":False}
    (Path("/kaggle/working")/"hala-h119-private-train-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2),"utf8")
    return receipt

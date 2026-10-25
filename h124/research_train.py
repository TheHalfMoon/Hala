"""H124 private new 896-source fresh joint START+END+Delta vision+language."""
import json,math,time
from pathlib import Path
from h106.research_train import config as parent_config
from h106.training_plan import STRATEGY as H106_SAMPLER
from h61.curriculum import load_export as load_h71
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h96.research_probe import require_private_vision_research
from h124.research_cpu import run as source_qualify
from h124.schedule import derive,TRAIN,HOLDOUT
from h124.training_plan import (STEPS,TRAIN_SHA,HELDOUT_SHA,
                                SCHEDULE_SHA,get_plan)
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.train_adapter import train
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

SCHEMA="hala.h124.fresh-original-complex-joint-start-end-delta896-T4.v1"

def config(repo):
    cfg=parent_config(Path(repo))
    if (cfg["max_steps"]!=256 or cfg["precision"]!="fp16"
            or cfg["batch_size"]!=1 or cfg["gradient_accumulation"]!=1
            or cfg["lora_rank"]!=8 or cfg["coordinate_weight"]!=8.0
            or cfg["supervision"]!="coordinate_weighted"
            or cfg["platform"]!="kaggle"):
        raise ValueError("H124 fresh noncommercial original GPU config drift")
    return {**cfg,"max_steps":STEPS}

def run(repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    require_private_vision_research(scratch=scratch,output=output)
    source_receipt=Path("/kaggle/working/hala-h124-gpu-original-source-preflight.json")
    if source_receipt.exists() or output.exists():
        raise ValueError("H124 create-only original private GPU files required")
    proof=source_qualify(scratch,source_receipt)
    if (proof["status"]!=
            "PASS_OWNER_H124_NEW_ORIGINAL_TRAIN1024_JOINT_GEOMETRY_SOURCE"
            or proof["train_private_sha256"]!=TRAIN_SHA
            or proof["heldout_private_sha256"]!=HELDOUT_SHA
            or proof["schedule_sha256"]!=SCHEDULE_SHA
            or proof["original_image_byte_and_two_box_SHA_verified"]!=1024
            or proof["all_previous_H105_H123_images_excluded"]!=1536):
        raise ValueError("H124 all 1024 true source START/END TRAIN checks missing")
    root=scratch/"H71/export"
    examples=root/"h61_drag_curriculum.jsonl"
    manifest=root/"h61_drag_curriculum_manifest.json"
    original=load_h71(examples,json.loads(manifest.read_text("utf8")))
    schedule=derive(original,root)
    plan=get_plan(schedule)
    if (len(plan.indices)!=TRAIN or len(schedule.heldout)!=HOLDOUT
            or plan.summary.get("no_heldout_optimizer_input") is not True):
        raise ValueError("H124 trained-vs-heldout original TRAIN overlap")
    params=config(repo)
    paramsfile=scratch/"h124-original-896-joint.json"
    paramsfile.write_text(json.dumps(params,sort_keys=True),"utf8")
    start=time.monotonic()
    train(str(paramsfile),str(examples),str(manifest),
          str(scratch/"H71/public-dev-baseline-receipt.json"),
          str(output),sampling_strategy=H106_SAMPLER,joint_lora=True,
          joint_geometry_mode=True)
    version,state=load_checkpoint(output,None)
    runtime=state.get("runtime",{})
    steps=runtime.get("qualification_steps",[])
    metrics=state.get("metrics",[])
    observed=[item.get("record_id") for st in steps
              for item in st.get("microbatches",[])]
    expected=[row["id"] for row in schedule.train]
    if (state.get("step")!=state.get("max_steps") or state.get("step")!=STEPS
            or len(steps)!=STEPS or len(metrics)!=STEPS
            or observed!=expected or len(set(observed))!=STEPS
            or runtime.get("h124_new_joint896_sampler",{}).get(
                "train_private_sha256")!=TRAIN_SHA
            or runtime.get("h124_new_joint896_sampler",{}).get(
                "heldout_private_sha256")!=HELDOUT_SHA
            or runtime.get("h124_new_joint896_sampler",{}).get(
                "schedule_sha256")!=SCHEDULE_SHA
            or runtime.get("h124_joint_original_processor_audits")!=12
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or len(runtime.get("h106_trainable_language_modules",[]))!=56
            or state.get("dataset_sha256")!=H71_SHA
            or state.get("changed_parameter_tensors",0)<=0):
        raise ValueError("H124 exact 896-step true paired-source optimizer receipt missing")
    for i,(step,metric) in enumerate(zip(steps,metrics),1):
        gradient=step.get("gradient_norm_before_clip")
        if (step.get("step")!=i or metric.get("step")!=i
                or step.get("changed_parameter_tensors_this_step",0)<=0
                or type(gradient) not in (int,float)
                or not math.isfinite(gradient) or gradient<=0
                or type(metric.get("loss")) not in (int,float)
                or not math.isfinite(metric["loss"]) or metric["loss"]<=0):
            raise ValueError("H124 genuine nonzero optimizer gradient at each step")
    report={
        "schema":SCHEMA,
        "status":"H124_REAL896_FRESH_JOINT_TRAIN_ONLY_OPTIMIZER_VERIFIED",
        "original_H71_TRAIN_SHA":H71_SHA,
        "train_private_sha256":TRAIN_SHA,
        "heldout_private_sha256":HELDOUT_SHA,
        "schedule_sha256":SCHEDULE_SHA,
        "original_source_media_and_dual_bbox_verified":1024,
        "actual_optimizer_steps":STEPS,"positive_gradient_steps":STEPS,
        "fresh_original_training_images":TRAIN,
        "unoptimized_new_heldout_images":HOLDOUT,
        "old_H105_H123_images_excluded":1536,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "final_adapter_sha256":sha256_file(version/"adapter/adapter_model.safetensors"),
        "final_checkpoint_sha256":sha256_file(version/"checkpoint.json"),
        "final_optimizer_sha256":sha256_file(version/"resume.pt"),
        "final_checkpoint_version":version.name,
        "initial_loss":metrics[0]["loss"],"last_loss":metrics[-1]["loss"],
        "elapsed_seconds":round(time.monotonic()-start,2),
        "original_EVAL_or_TEST_in_training":False,
        "heldout_used_for_optimizer":False,
        "prior_H113_H123_images_reused":False,
        "model_GUI_skill_qualified":False,
        "weights_public":False,"paid_compute_used":False}
    Path("/kaggle/working/hala-h124-private-training-receipt.json").write_text(
        json.dumps(report,sort_keys=True,indent=2),"utf8")
    return report

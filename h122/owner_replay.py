"""Independent owner Windows H122 448-step and 64 TRAIN-heldout replay.

Run only against downloaded PRIVATE original H71 TRAIN source, exact H113
source rows and actual Kaggle checkpoint + heldout raw outputs. Nothing
here sends data over network, trains, evaluates public EVAL, or publishes
models. Run this *separately* from the model-producing Kaggle notebook.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from h61.curriculum import load_export as load_h71
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h122.schedule import qualify_owner_private,TRAIN,HOLDOUT
from h122.heldout_source import admit_original_train_heldout
from h122.admit_checkpoint import (expected_training_identity,
                                    H122_VERIFIED_ADAPTER_SHA)
from h122.heldout_gate import score_heldout
from h122.research_train import (
    TRAIN_SHA_H122,HELDOUT_SHA_H122,SCHEDULE_SHA_H122)
from h122.training_plan import STEPS
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint

RAW_NAME="h122-private-original-train64-neural.json"
CASES_NAME="h122-private-original-train64-source-cases.json"
SUMMARY_NAME="h122-private-original-train64-END-summary.json"


def read_json(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("H122 exact private local artifact missing")
    return json.loads(path.read_text("utf8"))


def audit(repo:Path,root:Path,points:Path,checkpoint:Path,run_dir:Path):
    repo,root,checkpoint,run_dir=map(Path,(repo,root,checkpoint,run_dir))
    schedule=qualify_owner_private(points)
    if (schedule.summary.get("train_private_sha256")!=TRAIN_SHA_H122
            or schedule.summary.get("heldout_private_sha256")!=HELDOUT_SHA_H122
            or schedule.summary.get("schedule_sha256")!=SCHEDULE_SHA_H122):
        raise ValueError("H122 exact original TRAIN 448/64 SHA drift")
    manifest=read_json(root/"h61_drag_curriculum_manifest.json")
    rows=load_h71(root/"h61_drag_curriculum.jsonl",manifest)
    cases=admit_original_train_heldout(rows,root,schedule)
    if len(cases)!=HOLDOUT or len(set(s["image_sha256"] for s in cases))!=HOLDOUT:
        raise ValueError("H122 owner original source media disjointness drift")
    training_receipt=read_json(checkpoint.parent/"hala-h122-private-train-receipt.json")
    version,state=load_checkpoint(checkpoint,None)
    files={"adapter":version/"adapter/adapter_model.safetensors",
           "checkpoint":version/"checkpoint.json",
           "optimizer":version/"resume.pt"}
    measured={k:sha256_file(path) for k,path in files.items()}
    expected={"adapter":training_receipt.get("final_adapter_sha256"),
              "checkpoint":training_receipt.get("final_checkpoint_sha256"),
              "optimizer":training_receipt.get("final_optimizer_sha256")}
    identity,plan=expected_training_identity(repo,rows,root,schedule)
    updates=state.get("runtime",{}).get("qualification_steps",[])
    source_ids=[b.get("record_id") for step in updates
                for b in step.get("microbatches",[])]
    model_checks={
       "actual_448_step_training_receipt":
           training_receipt.get("status")=="H122_REAL448_TRAIN_ONLY_END_GRID_OPTIMIZER_VERIFIED"
           and training_receipt.get("actual_optimizer_steps")==STEPS,
       "original_train_exact_source":
           state.get("dataset_sha256")==H71_SHA
           and training_receipt.get("source_original_h71_SHA")==H71_SHA,
       "original_train_448_heldout_64":
           training_receipt.get("original_train_screenshots")==TRAIN
           and training_receipt.get("original_heldout_train_screenshots")==HOLDOUT,
       "original_train_and_heldout_private_SHA":
           training_receipt.get("train_private_sha256")==TRAIN_SHA_H122
           and training_receipt.get("heldout_private_sha256")==HELDOUT_SHA_H122
           and training_receipt.get("schedule_sha256")==SCHEDULE_SHA_H122,
       "actual_checkpoint_step_count":
           state.get("step")==STEPS and state.get("max_steps")==STEPS
           and len(state.get("metrics",[]))==STEPS and len(updates)==STEPS,
       "exact_no_heldout_optimizer_microbatch":
           source_ids==[r["id"] for r in schedule.train]
           and len(set(source_ids))==STEPS
           and not set(source_ids)&{r["id"] for r in schedule.heldout},
       "frozen_exact_optimizer_code_identity":
           state.get("config_sha256")==identity,
       "actual_joint_visual_language_adapter":
           state.get("runtime",{}).get("h106_joint_lora_verified") is True
           and len(state.get("runtime",{}).get("h106_trainable_visual_modules",[]))==8
           and len(state.get("runtime",{}).get("h106_trainable_language_modules",[]))==56,
       "positive_real_optimizer_gradients_all_steps":
           all(step.get("step")==i and
               step.get("changed_parameter_tensors_this_step",0)>0 and
               type(step.get("gradient_norm_before_clip")) in (int,float) and
               math.isfinite(step["gradient_norm_before_clip"]) and
               step["gradient_norm_before_clip"]>0
               for i,step in enumerate(updates,1)),
       "all_three_original_binary_SHA_matches":
           all(type(expected[k]) is str and len(expected[k])==64 and
               measured[k]==expected[k] for k in measured),
       "training_never_saw_EVAL_or_heldout":
           training_receipt.get("no_EVAL_or_TEST_used") is True and
           training_receipt.get("actual_validation_scored") is False,
       "no_paid_compute_or_public_model":
           training_receipt.get("paid_compute") is False and
           training_receipt.get("model_weights_published") is False
    }
    failures=[key for key,v in model_checks.items() if not v]
    if failures:
        raise ValueError("H122 independent 448-step model readback failed: "+",".join(failures))

    summary=read_json(run_dir/SUMMARY_NAME)
    raw_path=run_dir/RAW_NAME
    cases_path=run_dir/CASES_NAME
    if (summary.get("status")!="H122_REAL_PRIVATE_ORIGINAL_TRAIN64_HELDOUT_SCORED"
            or summary.get("private_raw_model_sha256")!=sha256_file(raw_path)
            or summary.get("private_original_source_cases_sha256")!=sha256_file(cases_path)
            or summary.get("actual_model_inference_calls")!=HOLDOUT
            or summary.get("no_fresh_PUBLIC_EVAL_or_hidden_TEST_read") is not True):
        raise ValueError("H122 private original heldout model inference receipt incomplete")
    upstream_cases=read_json(cases_path)
    if len(upstream_cases)!=HOLDOUT:
        raise ValueError("H122 real private original 64 cases incomplete")
    for reference,original in zip(upstream_cases,cases):
        # Absolute Kaggle/Windows filesystem paths differ. Verify all
        # original task/point/media identities and geometry instead.
        for key in ("id","split","source_split","family","language",
                    "image_sha256","image_width","image_height","prompt","end_bbox"):
            if reference.get(key)!=original.get(key):
                raise ValueError("H122 Kaggle and Windows original TRAIN case mismatch")
    neural=read_json(raw_path)
    own=score_heldout(cases,neural,schedule)
    comparison=("strict_valid_END_grid_and_point","strict_END_hits",
                "unique_valid_END_points","distinct_valid_grid_cells",
                "modal_point_fraction","original_train_families",
                "hits_per_family","holdout_gate_passed","thresholds")
    for key in comparison:
        if own.get(key)!=summary.get(key):
            raise ValueError("H122 owner H71 source and raw neural score drift: "+key)
    return {
       "schema":"hala.h122.independent-owner-TRAIN64-END-replay.v1",
       "status":"PASS_OWNER_H122_REAL448_OPTIMIZER_AND_TRAIN64_NEURAL_REPLAY",
       "training_checks_passed":len(model_checks),
       "training_checks_total":len(model_checks),
       "original_train_steps":STEPS,
       "heldout_train_scenes":HOLDOUT,
       "genuine_model_inference_calls":HOLDOUT,
       "owner_measured_binary_SHA":measured,
       "original_train_digest":TRAIN_SHA_H122,
       "heldout_train_digest":HELDOUT_SHA_H122,
       "strict_END_hits":own["strict_END_hits"],
       "native_grid_point_valid":own["strict_valid_END_grid_and_point"],
       "modal_point_fraction":own["modal_point_fraction"],
       "holdout_gate_passed":own["holdout_gate_passed"],
       "original_EVAL_or_TEST_used":False,
       "external_GUI_skill_proven":False,
       "weights_uploaded_to_public":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","h71_root","h113_points","checkpoint_root",
                "heldout_run_dir","private_report"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    proof=audit(a.repo,a.h71_root,a.h113_points,
                a.checkpoint_root,a.heldout_run_dir)
    if a.private_report.exists():
        raise ValueError("independent private audit report already exists")
    a.private_report.parent.mkdir(parents=True,exist_ok=True)
    a.private_report.write_text(
        json.dumps(proof,sort_keys=True,indent=2)+"\n","utf8")
    print(json.dumps(proof,sort_keys=True,indent=2))

if __name__=="__main__":
    main()

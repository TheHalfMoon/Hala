"""Independent Windows H123 real TRAIN896 and original TRAIN128 neural replay.

Requires the actual owner-private H71 original archive restored locally,
H123 trained adapter/checkpoint/optimizer binaries and genuine Kaggle
neural responses. Do NOT use public EVAL/TEST or download model weights
into this source code; this file only validates private local receipts.
"""
import argparse,json,math
from pathlib import Path

from h61.curriculum import load_export as load_h71
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h123.schedule import derive,TRAIN,HOLDOUT
from h123.training_plan import TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA,STEPS
from h123.admit_checkpoint import expected_identity
from h123.heldout_source import admit_original_train128
from h123.heldout_gate import score
from h123.eval_heldout import RAW,CASES,SUMMARY
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file


def read(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("H123 original owner-private evidence file missing")
    return json.loads(path.read_text("utf8"))


def audit(repo,root,checkpoint,heldout):
    root,checkpoint,heldout=map(Path,(root,checkpoint,heldout))
    originals=load_h71(root/"h61_drag_curriculum.jsonl",
                       read(root/"h61_drag_curriculum_manifest.json"))
    schedule=derive(originals,root)
    cases=admit_original_train128(originals,root,schedule)
    train_receipt=read(checkpoint.parent/"hala-h123-private-training-receipt.json")
    version,state=load_checkpoint(checkpoint,None)
    paths={"adapter":version/"adapter/adapter_model.safetensors",
           "checkpoint":version/"checkpoint.json",
           "optimizer":version/"resume.pt"}
    measured={k:sha256_file(v) for k,v in paths.items()}
    signature,plan=expected_identity(repo,schedule)
    runtime=state.get("runtime",{})
    steps=runtime.get("qualification_steps",[])
    rows=[v.get("record_id") for step in steps
          for v in step.get("microbatches",[])]
    proof={
      "true_896_train_optimizer_receipt":
          train_receipt.get("status")==
          "H123_REAL896_NEW_COMPLEX_TRAIN_ONLY_OPTIMIZER_VERIFIED"
          and train_receipt.get("actual_train_optimizer_steps")==STEPS,
      "actual_896_step_checkpoint_and_metrics":
          state.get("step")==state.get("max_steps")==STEPS
          and len(steps)==len(state.get("metrics",[]))==STEPS,
      "source_original_H71_dataset_digest":
          state.get("dataset_sha256")==H71_SHA==
          train_receipt.get("original_H71_TRAIN_SHA"),
      "SHA_frozen_original_896_128_screenshots":
          schedule.summary.get("train_private_sha256")==TRAIN_SHA
          and schedule.summary.get("heldout_private_sha256")==HELDOUT_SHA
          and schedule.summary.get("schedule_sha256")==SCHEDULE_SHA
          and len(cases)==HOLDOUT,
      "real_896_train_IDs_never_include_heldout":
          rows==[x["id"] for x in schedule.train]
          and len(set(rows))==TRAIN
          and not set(rows)&{r["id"] for r in schedule.heldout},
      "source_true_1024_image_SHA_rechecked":
          train_receipt.get("original_source_media_SHA_verified")==1024,
      "frozen_exact_training_config_and_sampler":
          state.get("config_sha256")==signature
          and runtime.get("h123_complex896_original_plan",{}).get(
              "plan_sha256")==plan.sha256,
      "actual_joint_vision_language_model_and_processor":
          runtime.get("h106_joint_lora_verified") is True
          and len(runtime.get("h106_trainable_visual_modules",[]))==8
          and len(runtime.get("h106_trainable_language_modules",[]))==56
          and runtime.get("h123_original_complex_processor_audits")==12,
      "positive_896_gradient_updates":
          all(t.get("step")==i
             and t.get("changed_parameter_tensors_this_step",0)>0
             and type(t.get("gradient_norm_before_clip")) in (int,float)
             and math.isfinite(t["gradient_norm_before_clip"])
             and t["gradient_norm_before_clip"]>0
             for i,t in enumerate(steps,1)),
      "real_all_three_binary_SHA_matches":
          all(measured[k]==train_receipt.get("final_"+k+"_sha256")
              for k in paths),
      "no_prior_H113_H122_EVAL_or_heldout_training":
          train_receipt.get("previous_H113_H122_source_reused") is False
          and train_receipt.get("original_EVAL_or_TEST_in_training") is False
          and train_receipt.get("heldout_scored_or_used_for_optimizer") is False,
      "no_paid_resources_or_published_model":
          train_receipt.get("paid_compute_used") is False
          and train_receipt.get("model_weights_public") is False,
    }
    failed=[k for k,v in proof.items() if not v]
    if failed:
        raise ValueError("H123 independent optimizer/adapter source proof FAILED: "+
                         ",".join(failed))
    summary=read(heldout/SUMMARY)
    raw_path=heldout/RAW
    cases_path=heldout/CASES
    if (summary.get("status")!=
            "H123_REAL_PRIVATE_ORIGINAL_TRAIN128_HELDOUT_SCORED"
            or summary.get("actual_model_inference_calls")!=HOLDOUT
            or summary.get("private_raw_neural_sha256")!=sha256_file(raw_path)
            or summary.get("private_original_cases_sha256")!=sha256_file(cases_path)):
        raise ValueError("H123 true private original TRAIN128 inference receipt invalid")
    upcases=read(cases_path)
    if len(upcases)!=HOLDOUT:
        raise ValueError("H123 exactly 128 private neural original source cases required")
    for own,reference in zip(cases,upcases):
        # Kaggle and Windows absolute paths differ, do not compare image_path.
        for key in ("id","split","source_split","family","language",
                    "image_sha256","image_width","image_height","prompt","end_bbox"):
            if own.get(key)!=reference.get(key):
                raise ValueError("H123 Kaggle source/END box differs from original Windows H71")
    predictions=read(raw_path)
    scored=score(cases,predictions,schedule)
    for key in ("strict_valid_END_dualgrid_and_point",
                "strict_END_hits","hits_per_family",
                "source_family_counts","unique_native_normalized_END_points",
                "modal_normalized_point_fraction","thresholds",
                "heldout_gate_passed","schedule_sha256"):
        if scored.get(key)!=summary.get(key):
            raise ValueError("H123 private neural result differs from Windows source scorer: "+key)
    return {
        "schema":"hala.h123.owner-independent-TRAIN896-TRAIN128-neural-replay.v1",
        "status":"PASS_OWNER_H123_TRUE_MODEL_AND_ORIGINAL_TRAIN128_REPLAY",
        "training_checks_passed":len(proof),
        "training_checks_total":len(proof),
        "actual_train_steps":STEPS,"original_heldout_scenes":HOLDOUT,
        "real_neural_model_calls":HOLDOUT,
        "original_source_media_rebuilt":True,
        "model_SHA256":measured,
        "strict_END_hits":scored["strict_END_hits"],
        "hits_per_family":scored["hits_per_family"],
        "heldout_gate_passed":scored["heldout_gate_passed"],
        "no_EVAL_or_TEST":True,
        "external_GUI_skill_proven":False,
        "public_weights":False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","h71_root","checkpoint_root","heldout_run_dir","private_report"):
        ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=ap.parse_args()
    out=audit(a.repo,a.h71_root,a.checkpoint_root,a.heldout_run_dir)
    if a.private_report.exists():
        raise ValueError("H123 private independent report exists: no overwrite")
    a.private_report.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n","utf8")
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=="__main__":
    main()

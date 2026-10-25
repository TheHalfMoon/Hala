"""Hala H123 owner-only free Kaggle CPU original TRAIN 1024 source qualifier.

This notebook does not train, evaluate models, read original EVAL/TEST,
or publish original image bytes, bounding boxes or private chosen IDs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="3a95597d3eef6e908f7ddf44a58d6a75864628d8"
REPO=Path("/kaggle/tmp/hala-h123-source-sha")
WORK=Path("/kaggle/tmp/hala-h123-original-10k-train-only")
OUTPUT=Path("/kaggle/working/hala-h123-private-unseen1024-source-receipt.json")
ATTEMPT=Path("/kaggle/working/hala-h123-source-attempt.json")


def record(x):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(x,indent=2,sort_keys=True),"utf8")
    tmp.replace(ATTEMPT)


def run_checked(*args,timeout=180):
    subprocess.run(list(args),check=True,capture_output=True,text=True,
                   timeout=timeout)


def main():
    state={"schema":"hala.h123.private-CPU-source-1024-original-TRAIN.v1",
       "source_git_sha":SOURCE_SHA,"status":"UNQUALIFIED",
       "stage":"OWNER_NONCOMMERCIAL_FREE_CPU",
       "actual_model_training_or_test_calls":0,
       "model_weights_published":False,
       "original_EVAL_or_TEST_used":False,
       "paid_compute_used":False,
       "commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H123 owner-private Kaggle only")
        # Only import H73 controls from the exact SHA-frozen source.
        if REPO.exists():
            raise ValueError("H123 preexisting source checkout rejected")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        run_checked("git","init","-q",str(REPO))
        run_checked("git","-C",str(REPO),"remote","add","origin",
                    "https://github.com/TheHalfMoon/Hala.git")
        run_checked("git","-C",str(REPO),"fetch","--depth","1",
                    "origin",SOURCE_SHA,timeout=180)
        run_checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        actual=subprocess.check_output(["git","-C",str(REPO),
                                        "rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("H123 frozen source commit does not match")
        sys.path.insert(0,str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        state["stage"]="EXACT_H71_SOURCE_AND_ORIGINAL_FAMILY_SPATIAL_SELECTION"
        record(state)
        from h123.research_cpu import run
        report=run(WORK,OUTPUT)
        if (report.get("status")!=
                "PASS_OWNER_H123_NEW_ORIGINAL_1024_COMPLEX_TRAIN_SOURCE"
                or report.get("train_source_images")!=896
                or report.get("heldout_source_images")!=128
                or report.get("image_bytes_and_original_END_box_SHA_verified")!=1024
                or report.get("no_prev_heldout_in_optimizer_or_new_holdout") is not True
                or report.get("no_original_PUBLIC_EVAL_or_TEST") is not True):
            raise ValueError("H123 genuine new 896/128 source proof failed")
        state.update(status="PASS_H123_CPU_SOURCE_ONLY",
                     stage="PRIVATE_OWNER_SOURCE_SHA_RECEIPT",
                     train_source_images=896,heldout_images=128,
                     private_schedule_sha=report["schedule_sha256"])
        record(state)
        print("H123_NEW_COMPLEX_SOURCE_ONLY_1024_QUALIFIED_NOT_MODEL",flush=True)
    except Exception as e:
        state.update(status="FAILED_H123_SOURCE_NOT_QUALIFIED",
                     failure_class=type(e).__name__)
        record(state)
        print("H123_SOURCE_REFUSED",type(e).__name__,flush=True)
        raise


if __name__=="__main__":
    main()

"""H113 PRIVATE CPU-only original H71 TRAIN source-qualified 512-pair audit.

This script never allocates a GPU, model, optimizer or paid service.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="ed615c83cf2e316abbf515e6edf32cf52ca47f63"
REPO=Path("/kaggle/tmp/hala-h113-source")
SCRATCH=Path("/kaggle/tmp/hala-h113-original-h71-train")
OUTPUT=Path("/kaggle/working/hala-h113-private-source-bound-dual-point")
ATTEMPT=Path("/kaggle/working/hala-h113-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temporary=ATTEMPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    temporary.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)


def main():
    status={"schema":"hala.h113.private-original-H71-train-dual-point-CPU.v1",
            "status":"STARTED","stage":"OWNER_PRIVATE_CPU_SOURCE_PREFLIGHT",
            "pinned_source_sha":SOURCE_SHA,
            "model_training_executed":False,
            "gpu_requested":False,"model_weights_published":False,
            "paid_resources_used":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("must use original owner's private Kaggle CPU session")
        if os.environ.get("CUDA_VISIBLE_DEVICES","").strip() not in ("","-1"):
            raise ValueError("H113 CPU-only admission forbids CUDA devices")
        status["stage"]="PINNED_ORIGINAL_TRAIN_SOURCE_CODE"
        record(status)
        if REPO.exists():
            raise ValueError("preexisting source workspace not trusted")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=35)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("pinned H113 original code SHA drift")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        status["stage"]="ORIGINAL_H71_SOURCE_ARCHIVE_AND_H105_512_QUANTIZATION_AUDIT"
        record(status)
        from h113.audit import run
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result["status"]!="ORIGINAL_H71_TRAIN_ONLY_DUAL_POINT_SOURCE_AUDITED"
                or result["verified_original_source_screenshots_considered"]!=512
                or result["source_split"]!="TRAIN_ONLY"
                or result["no_eval_or_test_sources"] is not True
                or result["model_training_performed"] is not False):
            raise ValueError("H113 original source grounded private receipt invalid")
        status.update(status=result["status"],
                      stage="PRIVATE_TRAIN_SOURCE_SFT_DATA_AND_ANONYMOUS_DIGEST_SAVED",
                      admitted_images=result["qualified_original_unique_images"],
                      independent_point_completions=result["qualified_independent_point_completion_count"],
                      quantization_quarantine=result["quarantine_counts"],
                      private_sft_sha256=result["derived_private_point_sft_sha256"])
        record(status)
        print("H113_OWNER_REAL_H71_TRAIN_ONLY_DUAL_POINT_AUDIT_COMPLETE",flush=True)
    except Exception as exc:
        status.update(status="FAILED_H113_NO_NEW_QUALIFIED_PRIVATE_POINT_DATA",
                      failure_class=type(exc).__name__)
        record(status)
        print("H113_PRIVATE_TRAIN_ONLY_DATA_NOT_QUALIFIED",type(exc).__name__,flush=True)
        raise


if __name__=="__main__":main()

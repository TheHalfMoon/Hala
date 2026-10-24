"""H105 owner-private free Kaggle CPU original H71 candidate admission only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_SHA="ca459cf0d4ac28d84aa1b1fc69484a811129dc02"
REPO=Path("/kaggle/tmp/hala-h105-source")
SCRATCH=Path("/kaggle/tmp/hala-h105-source-original-h71")
OUTPUT=Path("/kaggle/working/hala-h105-private-curriculum-512")
ATTEMPT=Path("/kaggle/working/hala-h105-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,text=True,capture_output=True,timeout=timeout)


def main():
    status={"schema":"hala.h105.private-original-512-cpu-admission.v1",
            "source_sha":SOURCE_SHA,"status":"STARTED","stage":"PRIVATE_CPU_PREFLIGHT",
            "model_weights_downloaded":False,"GPU_used":False,
            "optimizer_steps":0,"training_executed":False,
            "commercial_use":False,"public_data_upload":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private noncommercial Kaggle runtime required")
        Path("/kaggle/tmp").mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage("/kaggle/tmp").free<5*1024**3:
            raise OSError("insufficient private original H71 CPU scratch")
        status["stage"]="PINNED_PRIVATE_GIT_SOURCE"
        record(status)
        if REPO.exists():raise ValueError("source checkout must be new")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=45)
        sha=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if sha!=SOURCE_SHA:raise ValueError("source commit drift")
        sys.path.insert(0,str(REPO))
        status["stage"]="ORIGINAL_H71_10K_SOURCE_512_GEOMETRY_DIVERSITY"
        record(status)
        from h105.source_run import run
        out=run(SCRATCH,OUTPUT)
        status.update(status=out["status"],stage="PRIVATE_CPU_AGGREGATE_RECEIPT",
                      original_source_512_media_oracle_checks=out["original_source_screenshot_SHA_and_strict_oracle_verified"],
                      new_dominant_direction_fraction=out["proposed_dominant_direction_fraction"])
        record(status)
        if out["status"]!="QUALIFIED_512_PROPOSED_TRAIN_GEOMETRY_COVERAGE_NOT_TRAINED":
            raise ValueError("original H71 did not pass all pre-registered source diversity gates")
        print("H105_512_SOURCE_BOUND_TRAIN_CURRICULUM_READY_NOT_TRAINED",flush=True)
    except Exception as error:
        status["status"]="FAILED_H105_ACTUAL_SOURCE_COVERAGE_QUALIFICATION"
        status["failure_class"]=type(error).__name__
        record(status)
        print("H105_SOURCE_COVERAGE_BLOCKED",type(error).__name__,flush=True)
        raise


if __name__=="__main__":main()

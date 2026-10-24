"""H108 private free T4 original-vs-ruler 32-scene paired ablation runner."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="cae55871f9838a7773a03992f3f76c53e9917530"
REPO=Path("/kaggle/tmp/hala-h108-source")
WORK=Path("/kaggle/tmp/hala-h108-source-eval36-43")
OUTPUT=Path("/kaggle/working/hala-h108-private-original-vs-ruler-ab")
ATTEMPT=Path("/kaggle/working/hala-h108-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    temp.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,text=True,capture_output=True,timeout=timeout)


def main():
    status={"schema":"hala.h108.private-new-source-ruler-ab-attempt.v1",
            "stage":"OWNER_NONCOMMERCIAL_FREE_T4_PREFLIGHT","status":"STARTED",
            "pinned_source_SHA":SOURCE_SHA,
            "training_executed":False,"weights_published":False,
            "paid_cloud":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner private Kaggle research notebook required")
        import torch
        if not torch.cuda.is_available() or not any(
           "T4" in torch.cuda.get_device_name(i)
           for i in range(torch.cuda.device_count())):
            raise ValueError("free Kaggle T4 accelerator required")
        status["stage"]="PINNED_NO_PAID_GPU_PROCESSOR_DEPENDENCIES"
        record(status)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0]
                for k in expected}!=expected:
            raise ValueError("H108 original pinned processor dependency drift")
        status["stage"]="PINNED_GITHUB_SOURCE_FETCH"
        record(status)
        if REPO.exists():raise ValueError("unexpected preexisting source")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=45)
        current=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if current!=SOURCE_SHA:
            raise ValueError("H108 immutable source mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        status["stage"]="FROZEN_FOUR_ARM_NEW_SOURCE_PIXEL_RULER_EVALUATION"
        record(status)
        from h108.eval import run,ARMS
        result=run(repo=REPO,work=WORK,output=OUTPUT)
        if (result["status"]!="REAL_H108_ORIGINAL_VS_EXPLICIT_RULER_32_SOURCE_AB"
                or result["sample_counts"]!={arm:32 for arm in ARMS}
                or result["unaltered_original_source_bbox_scorer"] is not True
                or result["no_model_weights_modified"] is not True):
            raise ValueError("original strict source benchmark incomplete")
        status.update(status="REAL_H108_PRIVATE_SOURCE_PIXEL_RULER_AB_COMPLETE",
                      stage="OWNER_PRIVATE_NEURAL_ATTEMPT_RECEIPT",
                      paired_source_samples=32,
                      strictly_scored_actions=result["strict_successes"],
                      raw_attempts_sha256=result["raw_neural_attempts_sha256"])
        record(status)
        print("H108_REAL_ORIGINAL_SOURCE_RULER_AB_RECORDED",flush=True)
    except Exception as error:
        status["status"]="FAILED_H108_SOURCE_RULER_QUALIFICATION"
        status["failure_class"]=type(error).__name__
        record(status)
        print("H108_RESEARCH_NOT_QUALIFIED",type(error).__name__,flush=True)
        raise
if __name__=="__main__":main()

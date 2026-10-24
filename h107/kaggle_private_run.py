"""Owner-private Kaggle H107 frozen 64-scene paired GUI benchmark.

H106 pretrained checkpoint must be COMPLETE and SHA-qualified. This script
never trains, edits scorer, relaxes success criteria, or publishes weights.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="TO_BE_PINNED_AFTER_CODE_COMMIT"
REPO=Path("/kaggle/tmp/hala-h107-source")
WORK=Path("/kaggle/tmp/hala-h107-original-eval-unseen-20-to-35")
OUTPUT=Path("/kaggle/working/hala-h107-private-fresh-strict64")
ATTEMPT=Path("/kaggle/working/hala-h107-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    t=ATTEMPT.with_suffix(".tmp")
    t.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    t.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)


def main():
    status={"schema":"hala.h107.private-original-drag-64-matched-eval-attempt.v1",
            "stage":"OWNER_PRIVATE_T4_PREFLIGHT","status":"STARTED",
            "source_sha":SOURCE_SHA,"training_executed":False,
            "checkpoint_publication":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private Kaggle noncommercial experiment only")
        import torch
        if (not torch.cuda.is_available() or
                not any("T4" in torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count()))):
            raise ValueError("free Kaggle T4 evaluation required")
        status["stage"]="PINNED_PRIVATE_MODEL_DEPENDENCIES"
        record(status)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {x:importlib.metadata.version(x).split("+")[0] for x in expected}!=expected:
            raise ValueError("exact H107 model-dependency qualification failed")
        status["stage"]="PINNED_SOURCE_FETCH"
        record(status)
        if REPO.exists():raise ValueError("preexisting source workspace untrusted")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=30)
        got=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if got!=SOURCE_SHA:raise ValueError("H107 exact code source SHA changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        status["stage"]="SOURCE_SHA_BOUND_THREE_ARM_STRICT_GUI_EVAL_64"
        record(status)
        from h107.eval import run,ARMS
        report=run(repo=REPO,work=WORK,output=OUTPUT)
        if (report["status"]!="REAL_H107_FRESH_PAIRED_64_SOURCE_STRICT_EVAL_COMPLETE"
                or report["sample_counts"]!={arm:64 for arm in ARMS}
                or report["original_H61_scorer_modified"] is not False):
            raise ValueError("H107 paired 64-scene strict result incomplete")
        status.update(status="REAL_H107_FRESH_64_THREE_ARM_STRICT_EVAL_COMPLETE",
                      stage="OWNER_PRIVATE_RAW_NEURAL_ATTEMPTS_RECORDED",
                      paired_scene_counts=report["sample_counts"],
                      strict_GUI_successes=report["strict_successes"],
                      raw_sha256=report["raw_neural_attempts_sha256"])
        record(status)
        print("H107_REAL_64_SOURCE_THREE_ARM_STRICT_EVAL_COMPLETE",flush=True)
    except Exception as error:
        status.update(status="FAILED_H107_NO_NEW_QUALIFIED_GUI_EVAL",
                      failure_class=type(error).__name__)
        record(status)
        print("H107_EVALUATION_NOT_QUALIFIED",type(error).__name__,flush=True)
        raise


if __name__=="__main__":main()

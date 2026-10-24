"""H101 private free Kaggle T4 four-arm prompt-only control."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="4c29a309c549a4cf6ca4d78f143d78e7f5d25d5c"
REPO=Path("/kaggle/tmp/hala-h101-source")
SCRATCH=Path("/kaggle/tmp/hala-h101-fresh-original-drag-on")
OUTPUT=Path("/kaggle/working/hala-h101-private-eval")
ATTEMPT=Path("/kaggle/working/hala-h101-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,timeout=timeout,check=True)

def main():
    data={"schema":"hala.h101.private-coordinate-prompt-controlled-attempt.v1",
          "status":"STARTED","stage":"GPU_PREFLIGHT","source_sha":SOURCE_SHA,
          "training_executed":False,"model_published":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("owner-private Kaggle experiment required")
        import torch
        if not torch.cuda.is_available() or not any("T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("genuine free Kaggle Tesla T4 required")
        data["gpu_names"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        data["stage"]="PINNED_MODEL_DEPENDENCIES"
        record(data)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=480)
        pins={"torch":"2.11.0","transformers":"5.16.1",
              "peft":"0.20.0","Pillow":"11.3.0"}
        installed={k:importlib.metadata.version(k) for k in pins}
        if any(installed[k].split("+")[0]!=v for k,v in pins.items()):
            raise ValueError("runtime dependency drift")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("torchao unexpectedly installed")
        data["stage"]="PINNED_GIT_SOURCE"
        record(data)
        if REPO.exists():
            raise ValueError("preexisting unverified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=30)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("H101 source SHA mismatch")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        sys.path.insert(0,str(REPO))
        from h101.eval import run, ARMS
        data["stage"]="FOUR_ARM_FRESH_STRICT_UNSEEN_PROMPT_EVAL"
        record(data)
        results=run(repo=REPO,work=SCRATCH,output=OUTPUT)
        if (results.get("status")!="REAL_H101_CONTROLLED_NEW_UNSEEN_PROMPT_GEOMETRY_EVAL"
                or results.get("sample_counts")!={arm:16 for arm in ARMS}
                or not results.get("raw_evidence_sha256")
                or results.get("training_performed_during_evaluation") is not False):
            raise ValueError("no complete paired strict H101 evaluation")
        data.update(status="H101_NEW_PROMPT_AB_EVAL_COMPLETE",
                    stage="PRIVATE_RAW_SHA_BOUND",
                    strict_successes=results["strict_successes"],
                    raw_evidence_sha256=results["raw_evidence_sha256"])
        record(data)
        print("H101_REAL_FRESH_CONTROLLED_STRICT_EVAL_COMPLETE",flush=True)
    except Exception as error:
        data.update(status="FAILED_NO_H101_QUALIFIED_GUI_EVAL",
                    failure_class=type(error).__name__)
        record(data)
        print("H101_PRIVATE_EXPERIMENT_FAILED",type(error).__name__,flush=True)
        raise
if __name__=="__main__":
    main()

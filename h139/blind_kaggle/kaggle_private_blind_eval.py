"""Private one-shot H139 out-of-distribution model-only inference on free T4.

Never mount a TRAIN/DEV dataset or run an optimizer in this kernel.
All gold original-app coordinates stay owner-local on Windows.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="d4ac96e2e5f208be3b41c13b7644ba742efb870e"
REPO=Path("/kaggle/tmp/hala-h139-frozen-blind-inference-source")
OUT=Path("/kaggle/working/hala-h139-private-blind-native-output")
RECEIPT=Path("/kaggle/working/hala-h139-private-blind-run-attempt.json")

def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf8")
    temp.replace(RECEIPT)

def run(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H139 frozen blind private inference prerequisite failed")
    return result.stdout.strip()

def main():
    receipt={"schema":"hala.h139.one-shot-private-blind-eval-attempt.v1",
             "status":"NOT_QUALIFIED","source_git_sha":SOURCE_SHA,
             "optimizer_steps":0,"expected_model_calls":32,
             "paid_compute_used":False,"main_35B_updated":False,
             "independent_private_blind_only":True}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H139 fresh private Kaggle inference run only")
        import torch
        if not torch.cuda.is_available() or not any(
                "T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("H139 FREE T4 required, no paid fallback")
        names=[p.name.lower() for p in Path("/kaggle/input").iterdir() if p.is_dir()]
        if any(f in n for n in names for f in ("train64","dev16","sealed")):
            raise ValueError("H139 only frozen blind/redacted and adapter datasets allowed")
        run(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        run(sys.executable,"-m","pip","install","-q",
            "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        observed={n:importlib.metadata.version(n).split("+")[0] for n in expected}
        if observed!=expected:
            raise ValueError("H139 frozen blinded runtime version drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        run("git","init","-q",str(REPO))
        run("git","-C",str(REPO),"remote","add","origin",
            "https://github.com/TheHalfMoon/Hala.git")
        run("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        run("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if run("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H139 immutable blind source Git ref drift")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h139.blind_gpu import run as neural_run
        receipt["status"]="RUNNING_32_FROZEN_BLIND_NEURAL_CALLS"
        save(receipt)
        result=neural_run(OUT)
        if (result.get("status")!="H139_COMPLETE_32_REAL_BLIND_NEURAL_CALLS" or
            result.get("model_calls")!=32 or
            result.get("optimizer_steps")!=0 or
            result.get("paid_compute_used") is not False):
            raise ValueError("H139 blinded neural output not qualified")
        receipt.update(result)
        save(receipt)
    except Exception as exc:
        receipt.update(status="FAIL_H139_BLIND_NEURAL_NOT_QUALIFIED",
                       failure_class=type(exc).__name__)
        save(receipt)
        raise

if __name__=="__main__":
    main()

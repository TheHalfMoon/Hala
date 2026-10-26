"""One-time private frozen adapter SEALED16 Kaggle FREE T4 evaluator."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="4a47ff8b62aa4e980ea0a68fe7b532c2a8ceddde"
REPO=Path("/kaggle/tmp/hala-h134-once-frozen-sealed-code")
OUTPUT=Path("/kaggle/working/hala-h134-private-sealed16-after-freeze")
RECEIPT=Path("/kaggle/working/hala-h134-private-sealed-attempt.json")

def write(obj):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n","utf8")
    tmp.replace(RECEIPT)

def checked(*args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError("H134 sealed evaluator pinned prerequisite unavailable")
    return p.stdout.strip()

def main():
    receipt={
       "schema":"hala.h134.private-frozen-sealed16-attempt.v1",
       "status":"NOT_QUALIFIED",
       "source_git_sha":SOURCE_SHA,
       "training_optimizer_steps":0,
       "expected_inference_calls":64,
       "sealed_holdout_owner_only":True,
       "paid_compute":False,
       "main_Hala_weights_unchanged":True,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H134 sealed once-only private fresh Kaggle kernel")
        import torch
        if not torch.cuda.is_available() or not any("T4" in torch.cuda.get_device_name(i)
                                   for i in range(torch.cuda.device_count())):
            raise ValueError("H134 sealed only FREE T4; no paid resource")
        root=Path("/kaggle/input")
        if not root.is_dir():raise ValueError("H134 Kaggle private data root absent")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("H134 frozen sealed GPU model runtime drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H134 frozen sealed source Git identity changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        receipt["stage"]="NO_TRAINING_ON_SEALED_ONLY_INFERENCE"
        write(receipt)
        from h134.sealed_gpu import run
        result=run(OUTPUT)
        if (result.get("status")!="H134_REAL_PRIVATE_ONE_TIME_SEALED16_SCORED" or
            result.get("real_neural_calls")!=64 or
            result.get("actual_optimizer_steps")!=0 or
            result.get("original_train_dev_scenes_in_eval")!=0 or
            result.get("main_Hala_flagship_weights_updated") is not False):
            raise ValueError("H134 one-time private sealed inference receipt failed")
        receipt.update(result)
        write(receipt)
    except Exception as exc:
        receipt.update(status="FAIL_H134_SEALED_FROZEN_EVAL_NOT_QUALIFIED",
                       failure_class=type(exc).__name__)
        write(receipt)
        raise
if __name__=="__main__": main()

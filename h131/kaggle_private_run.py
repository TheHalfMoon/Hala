"""Pinned private free Kaggle T4 execution for H131 ShowUI initial screenshots."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="e07bbdcf32542831795537c00e6ccc081d115f7b"
PHASE="initial"
REPO=Path("/kaggle/tmp/hala-h131-private-code")
OUTPUT=Path("/kaggle/working/hala-h131-private-neural-"+PHASE)
ATTEMPT=Path("/kaggle/working/hala-h131-private-"+PHASE+"-attempt.json")
def record(value):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    path=ATTEMPT.with_suffix(".tmp")
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n","utf8")
    path.replace(ATTEMPT)
def checked(*args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("H131 frozen code or free T4 runtime unavailable")
    return p.stdout.strip()
def main():
    state={"schema":"hala.h131.private-neural-browser-stage.v1",
           "phase":PHASE,"status":"NOT_QUALIFIED","source_git_sha":SOURCE_SHA,
           "paid_compute_used":False,"model_optimizer_steps":0,
           "private_model_outputs":True,"actual_browser_actions":0}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H131 requires new private Kaggle execution")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H131 only Kaggle free NVIDIA T4")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        required={"transformers":"5.16.1","torch":"2.11.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in required}!=required:
            raise ValueError("H131 exact pretrained GPU runtime drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin","https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H131 exact frozen Git SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h131.gpu_infer import run
        state.update(stage="PINNED_SHOWUI_REAL_SCREENSHOTS")
        record(state)
        result=run(OUTPUT,PHASE)
        if (result.get("status")!="PASS_H131_PRIVATE_NEURAL_POINTS_NOT_ACTIONS" or
            result.get("real_model_calls")!=8 or result.get("optimizer_steps")!=0):
            raise ValueError("H131 strict real model inference evidence incomplete")
        state.update(result)
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H131_PRIVATE_REAL_NEURAL_POINTS",
                     failure_class=type(exc).__name__)
        record(state)
        raise
if __name__=="__main__": main()

"""Frozen H133 owner-private zero-cost Kaggle 8-scene comparison runner."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="__PIN_CODE_SHA__"
ARM="__MODEL_ARM__"
REPO=Path("/kaggle/tmp/hala-h133-frozen-model-code")
OUTPUT=Path("/kaggle/working/hala-h133-private-"+ARM+"-8")
ATTEMPT=Path("/kaggle/working/hala-h133-"+ARM+"-attempt.json")
def write_receipt(item):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(item,sort_keys=True,indent=2)+"\n","utf8")
    tmp.replace(ATTEMPT)
def checked(*args,timeout=180):
    proc=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError("H133 frozen private free model prerequisite command failed")
    return proc.stdout.strip()
def main():
    state={"schema":"hala.h133.private-pretrained-model-run.v1",
       "arm":ARM,"status":"NOT_QUALIFIED",
       "code_sha":SOURCE_SHA,"optimizer_steps":0,"paid_compute":False,
       "source_is_private":True,"model_skill_qualified":False}
    try:
        if ARM not in ("showui","specialist") or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H133 fresh private paired arm execution only")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H133 FREE NVIDIA T4 only, no paid fallback")
        version="5.16.1" if ARM=="showui" else "5.17.0"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers=="+version,"Pillow==11.3.0",timeout=540)
        required={"transformers":version,"torch":"2.11.0","Pillow":"11.3.0"}
        if {name:importlib.metadata.version(name).split("+")[0]
            for name in required}!=required:
            raise ValueError("H133 pinned model package runtime changed")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H133 preregistered code SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h133.neural import run
        state["stage"]="FROZEN_PRIVATE_NEURAL_UNSEEN8"
        write_receipt(state)
        result=run(OUTPUT,ARM)
        if (result.get("status")!="H133_PRIVATE_REAL_NEURAL_UNSEEN8_COMPLETED_NOT_SCORED"
            or result.get("real_neural_calls")!=16 or
            result.get("trained") is not False or
            result.get("paid_compute_used") is not False):
            raise ValueError("H133 exact private 8-scene neural arm not qualified")
        state.update(result)
        write_receipt(state)
    except Exception as exc:
        state.update(status="FAIL_H133_PRETRAINED_UNSEEN8_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        write_receipt(state)
        raise
if __name__=="__main__": main()

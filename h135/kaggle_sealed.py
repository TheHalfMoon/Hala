"""H135 exactly once: new sealed18 private T4 pretrained/adapter comparison."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="__SEALED_COMMIT_SHA__"
REPO=Path("/kaggle/tmp/hala-h135-frozen-one-time-sealed-code")
OUTPUT=Path("/kaggle/working/hala-h135-private-sealed18-once")
ATTEMPT=Path("/kaggle/working/hala-h135-sealed-attempt.json")
def store(receipt):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
    tmp.replace(ATTEMPT)
def cmd(*args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError("H135 exact frozen sealed GPU prereq failed")
    return p.stdout.strip()
def main():
    state={"schema":"hala.h135.once-private-sealed18-eval-receipt.v1",
       "status":"NOT_QUALIFIED","git_sha":SOURCE_SHA,
       "optimizer_steps":0,"paid_gpu":False,"sealed_owner_private":True,
       "adaptation_allowed":False,"main_Hala_flagship_weights_unchanged":True}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H135 private fresh once-only Kaggle T4")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
            raise ValueError("H135 FREE NVIDIA T4 only, no paid fallback")
        cmd(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        cmd(sys.executable,"-m","pip","install","-q",
            "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        got={k:importlib.metadata.version(k).split("+")[0] for k in expected}
        if got!=expected:raise ValueError("H135 pinned sealed runtime not identical to pilot")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        cmd("git","init","-q",str(REPO))
        cmd("git","-C",str(REPO),"remote","add","origin",
            "https://github.com/TheHalfMoon/Hala.git")
        cmd("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        cmd("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if cmd("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H135 once-only frozen code git commit mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["status"]="RUNNING_PRIVATE_NONTRAINING_SEALED18"
        store(state)
        from h135.sealed_gpu import run
        actual=run(OUTPUT)
        if (actual.get("status")!="H135_PRIVATE_ONCE_SEALED18_72_NEURAL_SCORED" or
            actual.get("actual_neural_calls")!=72 or
            actual.get("optimizer_steps")!=0 or
            actual.get("no_pilot_data_attached") is not True):
            raise ValueError("H135 once-only sealed true model inference unqualified")
        state.update(actual);store(state)
    except Exception as exc:
        state.update(status="FAIL_H135_SEALED18_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        store(state)
        raise
if __name__=="__main__":main()

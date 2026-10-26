"""Frozen free Kaggle T4 research: 2B specialist visual grounding."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="4073da9615493e83d3b4dd0f03d44c5292f795ad"
ROOT=Path("/kaggle/tmp/hala-h132-specialist-git")
OUTPUT=Path("/kaggle/working/hala-h132-private-specialist-2b")
ATTEMPT=Path("/kaggle/working/hala-h132-attempt.json")
def write(value):
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n","utf8")
    tmp.replace(ATTEMPT)
def checked(*cmd,timeout=180):
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H132 frozen code / GPU setup command failed")
    return result.stdout.strip()
def main():
    state={"schema":"hala.h132.free-private-specialist2b-attempt.v1",
           "status":"NOT_QUALIFIED","source_commit":SOURCE_SHA,
           "optimizer_steps":0,"model_calls":0,"paid_compute":False,
           "is_Hala_finetuned_model":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or ROOT.exists():
            raise ValueError("H132 only new private Kaggle T4 owner run")
        import torch
        if not torch.cuda.is_available() or not any(
           "T4" in torch.cuda.get_device_name(i)
           for i in range(torch.cuda.device_count())):
            raise ValueError("H132 free T4 GPU only, no paid fallback")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","torch":"2.11.0","Pillow":"11.3.0"}
        if {name:importlib.metadata.version(name).split("+")[0]
            for name in expected}!=expected:
            raise ValueError("H132 installed model runtime differs from predeclared versions")
        ROOT.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(ROOT))
        checked("git","-C",str(ROOT),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(ROOT),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(ROOT),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(ROOT),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H132 immutable source SHA mismatch")
        sys.path.insert(0,str(ROOT))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h132.research_gpu import run
        state["stage"]="PINNED_PRETRAINED_SPECIALIST_REAL_VISUAL_INFERENCE"
        write(state)
        result=run(OUTPUT)
        if result.get("actual_neural_calls")!=8 or result.get("training_steps")!=0:
            raise ValueError("H132 expected real 8 model calls, no optimization")
        state.update(result)
        write(state)
    except Exception as exc:
        state.update(status="FAIL_H132_MODEL_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        write(state)
        raise
if __name__=="__main__":
    main()

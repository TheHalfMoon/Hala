"""H134 Hala research-adapter: 24 real LoRA optimizer steps on FREE T4.

Only own original synthetic screenshots, private TRAIN128 and DEV16.
The SEALED TEST dataset is deliberately omitted from Kaggle metadata.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="__PIN_SOURCE_SHA__"
REPO=Path("/kaggle/tmp/hala-h134-source-and-training")
OUTPUT=Path("/kaggle/working/hala-h134-private-gui-start-end-lora")
RECEIPT=Path("/kaggle/working/hala-h134-private-training-attempt.json")

def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    temp.replace(RECEIPT)
def checked(*args,timeout=180):
    item=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if item.returncode:
        raise RuntimeError("H134 predeclared private training prerequisite failure")
    return item.stdout.strip()
def main():
    state={
        "schema":"hala.h134.private-2b-language-lora-training-attempt.v1",
        "status":"NOT_QUALIFIED",
        "source_git_sha":SOURCE_SHA,
        "optimizer_steps":0,
        "expected_optimizer_steps":24,
        "paid_compute_used":False,
        "owner_private_dataset":True,
        "sealed_holdout_attached":False,
        "external_dataset_training":False,
        "main_Hala_35B_weights_updated":False,
        "release_qualified":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H134 owner-only fresh Kaggle private noncommercial session")
        import torch
        if not torch.cuda.is_available() or not any(
                "T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("H134 only FREE Kaggle Nvidia T4; no paid fallback")
        inputs=Path("/kaggle/input")
        slugs=sorted(p.name for p in inputs.iterdir() if p.is_dir())
        if (not all(any(s in x for x in slugs) for s in
                    ("hala-h134-private-train128","hala-h134-private-dev16")) or
            any("sealed" in x or "test" in x or "h133" in x or "h71" in x for x in slugs)):
            raise ValueError("H134 exactly owner-private TRAIN and DEV, no sealed/test/old")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        required={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        got={name:importlib.metadata.version(name).split("+")[0]
             for name in required}
        if got!=required:
            raise ValueError("H134 model-training package versions drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H134 immutable training code git SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["status"]="RUNNING_REAL_PRIVATE_H134_LANGUAGE_LORA"
        save(state)
        from h134.train_gpu import run
        result=run(OUTPUT)
        if (result.get("status")!="H134_REAL_24_OPTIMIZER_STEP_LANGUAGE_LORA_EXPERIMENT" or
            result.get("actual_optimizer_steps")!=24 or
            result.get("sealed_holdout_attached") is not False or
            result.get("main_Hala_flagship_weights_updated") is not False or
            result.get("paid_compute_used") is not False):
            raise ValueError("H134 real adapter training evidence incomplete")
        state.update(result)
        save(state)
    except Exception as exc:
        state.update(status="FAIL_H134_REAL_LORA_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        save(state)
        raise
if __name__=="__main__":
    main()

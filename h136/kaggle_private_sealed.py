"""Frozen H136 ONE-TIME sealed24 external-2B versus genuine 64-step LoRA.

PRIVATE free Kaggle T4. There are NO optimizers and NO TRAIN/DEV mounts.
Git source and adapter hashes must match original owner-side freeze.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="5aaf2b74acd3396c6746aa095df4f34248dda06d"
REPO=Path("/kaggle/tmp/hala-h136-private-frozen-sealed-source")
OUTPUT=Path("/kaggle/working/hala-h136-private-sealed24-frozen")
RECEIPT=Path("/kaggle/working/hala-h136-private-sealed24-attempt.json")

def receipt(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n","utf8")
    temp.replace(RECEIPT)

def cmd(*args,timeout=180):
    completed=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if completed.returncode:
        raise RuntimeError("H136 frozen sealed GPU prerequisite failed")
    return completed.stdout.strip()

def main():
    state={
        "schema":"hala.h136.private-sealed24-attempt.v1",
        "status":"NOT_QUALIFIED","frozen_git_sha":SOURCE_SHA,
        "actual_optimizer_steps":0,
        "no_training_or_dev_sources":True,
        "owner_private":True,"paid_compute_used":False,
        "main_Hala_35B_unchanged":True}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H136 only fresh private owner Kaggle kernel")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H136 no paid GPUs, FREE Nvidia T4 only")
        root=Path("/kaggle/input")
        if not root.is_dir():
            raise ValueError("H136 private sealed root missing")
        for manifest in root.rglob("manifest.json"):
            try:
                item=json.loads(manifest.read_text("utf8"))
            except (OSError,ValueError,TypeError,json.JSONDecodeError):
                continue
            if item.get("schema")=="hala.h136.owner-original-workstation-gui-scenes.v1" and item.get("split")!="sealed":
                raise ValueError("H136 no TRAIN/DEV original-source manifests in sealed kernel")
        cmd(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        cmd(sys.executable,"-m","pip","install","-q","transformers==5.17.0",
            "peft==0.21.2","Pillow==11.3.0",timeout=540)
        wanted={"transformers":"5.17.0","peft":"0.21.2",
                "torch":"2.11.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in wanted}!=wanted:
            raise ValueError("H136 frozen sealed model GPU runtime changed")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        cmd("git","init","-q",str(REPO))
        cmd("git","-C",str(REPO),"remote","add","origin",
            "https://github.com/TheHalfMoon/Hala.git")
        cmd("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        cmd("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if cmd("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H136 sealed Git code SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["status"]="RUNNING_FROZEN_SEALED24_REAL_NEURAL"
        receipt(state)
        from h136.sealed_gpu import run
        result=run(OUTPUT)
        if (result.get("status")!="H136_REAL_ONCE_ONLY_SEALED24_NEURAL_COMPLETE" or
            result.get("neural_inference_calls")!=96 or
            result.get("actual_optimizer_steps")!=0 or
            result.get("train_or_dev_images_attached") is not False or
            result.get("paid_compute_used") is not False):
            raise ValueError("H136 sealed24 genuine neural completeness not proven")
        state.update(result)
        receipt(state)
    except Exception as exc:
        state.update(status="FAIL_H136_ONCE_SEALED_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        receipt(state)
        raise

if __name__=="__main__":main()

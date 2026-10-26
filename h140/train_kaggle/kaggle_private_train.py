"""PRIVATE H140 owner noncommercial research, FREE Kaggle T4 only.

Fetch immutable GitHub source, admit ONLY SHA-pinned H140 TRAIN256 and DEV64,
then 128 actual external 2B LoRA steps. No BLIND, no Hala flagship 35B,
no published model weights, no paid compute or fallback.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="dcd42cdb0bc1296ee787ffe177a460169b887c37"
REPO=Path("/kaggle/tmp/hala-h140-private-research-source")
OUTPUT=Path("/kaggle/working/hala-h140-private-original-geometry-lora")
RECEIPT=Path("/kaggle/working/hala-h140-private-training-attempt.json")
APPROVED_MOUNTS={
    "hala-h140-private-train256-original",
    "hala-h140-private-dev64-original",
}

def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf8")
    tmp.replace(RECEIPT)

def checked(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H140 pinned PRIVATE free-T4 prerequisite failed")
    return result.stdout.strip()

def source_mounts(root):
    actual={p.name for p in root.iterdir() if p.is_dir()}
    if actual!=APPROVED_MOUNTS:
        raise ValueError("H140 EXACTLY owner TRAIN+DEV mount slugs; no BLIND/unknown datasets")
    if any(any(fragment in name.lower() for fragment in
               ("blind","sealed","ood","test")) for name in actual):
        raise ValueError("H140 one-time OOD not permitted as model training input")

def main():
    status={"schema":"hala.h140.owner-private-only-free-kaggle-training-attempt.v1",
            "status":"H140_GPU_TRAINING_NOT_STARTED",
            "source_git_sha":SOURCE_SHA,
            "actual_optimizer_steps":0,
            "required_optimizer_steps":128,
            "owner_source_train":256,
            "owner_source_dev":64,
            "real_blind_calls":0,
            "paid_compute_used":False,
            "cloud_model_published":False,
            "main_Hala_35B_updated":False}
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
            REPO.exists() or OUTPUT.exists()):
            raise ValueError("H140 fresh private noncommercial owner Kaggle GPU only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H140 FREE Nvidia Tesla T4 required; no paid fallback")
        input_root=Path("/kaggle/input")
        if not input_root.is_dir():
            raise ValueError("H140 private owner datasets not mounted")
        source_mounts(input_root)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        observed={name:importlib.metadata.version(name).split("+")[0]
                  for name in expected}
        if observed!=expected:
            raise ValueError("H140 frozen free-T4 package dependency drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H140 immutable checked-out source SHA drift")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h140 import train_gpu
        train_gpu.DATA_ROOT=input_root
        # Checks exact owner-frozen original PNG sources and SHA for both roles.
        train_gpu.private_splits()
        status["status"]="H140_RUNNING_GENUINE_FREE_T4_ORIGINAL_GUI_SFT"
        save(status)
        result=train_gpu.run(OUTPUT)
        if (result.get("status")!="H140_COMPLETED_128_REAL_GPU_OPTIMIZER_STEPS"
            or result.get("real_optimizer_steps")!=128
            or result.get("real_blind_neural_calls")!=0
            or result.get("paid_compute_used") is not False
            or result.get("flagship_Hala_35B_updated") is not False):
            raise ValueError("H140 real original GUI adapter cannot be promoted")
        status.update(result)
        save(status)
    except Exception as exc:
        status.update(status="FAIL_H140_ORIGINAL_GPU_TRAINING_NOT_QUALIFIED",
                      failure_class=type(exc).__name__)
        save(status)
        raise

if __name__=="__main__":
    main()

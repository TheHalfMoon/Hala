"""Owner-private free Kaggle T4 H156 Qwen3.5-2B typed-action LoRA pilot."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="115493450870b8ba6e27cbe34634e3e6c65d03f4"
SOURCE_SHA="fe274bc48225685ea135137e1607169ef9481903c5848f9bb1e2ede78f509562"
WORK=Path("/kaggle/tmp/hala-h156-typed-action-research-source")
OUT=Path("/kaggle/working/hala-h156-typed-action-lora")
RECEIPT=Path("/kaggle/working/hala-h156-private-train-receipt.json")
DATASET_SLUG="hala-h156-private-typed-action-pilot-v1"


def checked(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,
                          check=False,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H156 Kaggle exact free-only prerequisite failed")
    return result.stdout.strip()


def store(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    path=RECEIPT.with_suffix(".tmp")
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    path.replace(RECEIPT)


def find_file(root,name):
    matches=[p for p in root.rglob(name) if p.is_file()]
    if len(matches)!=1:
        raise ValueError("H156 exactly one private dataset archive required")
    return matches[0]


def main():
    receipt={
        "schema":"hala.h156.free-private-t4-typed-gui-training-receipt.v1",
        "status":"H156_NO_TRAINING_YET",
        "source_commit":SOURCE_COMMIT,
        "source_manifest_sha256":SOURCE_SHA,
        "required_optimizer_steps":24,"optimizer_steps":0,
        "paid_gpu_used":False,"h140_sealed_blind_calls":0,
        "Hala_flagship_35B_updated":False,
        "model_weights_published":False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
            WORK.exists() or OUT.exists()):
            raise ValueError("H156 only fresh private Kaggle worker")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H156 free Nvidia T4 unavailable; no paid fallback")
        source_root=Path("/kaggle/input")
        if not source_root.is_dir():
            raise ValueError("H156 private dataset missing")
        if any(any(term in folder.name.lower() for term in
                   ("blind","h140","ood","h153","h154","h155"))
               for folder in source_root.iterdir()):
            raise ValueError("H156 unrelated holdout/private dataset mounted")
        images=find_file(source_root,"h156-source-images.zip")
        adapter=find_file(source_root,"h142-private-visible-column-lora.zip")
        if (DATASET_SLUG not in str(images) or
            images.parent!=adapter.parent):
            raise ValueError("H156 requires one owner-private source dataset")
        checked(sys.executable,"-m","pip","uninstall","-y",
                "torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        actual={name:importlib.metadata.version(name).split("+")[0]
                for name in ("torch","transformers","peft","Pillow")}
        if actual!={"torch":"2.11.0","transformers":"5.17.0",
                    "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H156 reproducible GPU dependency pin mismatch")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(WORK))
        checked("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        checked("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H156 pinned original source code changed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",
                          DO_NOT_TRACK="1",HALA_NONCOMMERCIAL_RESEARCH="1")
        from h156.train_gpu import run
        receipt["status"]="H156_RUNNING_NEW_TYPED_GUI_RESEARCH_OPTIMIZATION"
        store(receipt)
        result=run(images,adapter,OUT)
        if (result.get("status")!="H156_COMPLETED_REAL_TYPED_ACTION_OPTIMIZER_STEPS"
            or result.get("actual_optimizer_steps")!=24 or
            result.get("h140_sealed_blind_calls")!=0 or
            result.get("Hala_flagship_35B_updated") is not False):
            raise ValueError("H156 genuine GPU evidence failed")
        receipt.update(status="H156_REAL_24_FREE_T4_OPTIMIZER_STEPS_COMPLETE",
                       optimizer_steps=24,
                       updated_action_adapter_sha256=result["updated_action_adapter_sha256"],
                       dev_before=result["dev_before"],
                       dev_after=result["dev_after"])
        store(receipt)
    except Exception as exc:
        receipt.update(status="FAIL_H156_TRAINING_NOT_QUALIFIED",
                       failure_class=type(exc).__name__)
        store(receipt)
        raise


if __name__=="__main__":
    main()

"""H142 new original visible-heading bilingual LoRA, owner PRIVATE FREE T4.

Immutable source SHA and original trained-source manifest checks, 64 exact
real optimizer steps, 192 original role targets, NO H140/H141 neural
evaluation pictures, NO sealed BLIND, NO Hala 35B model changes.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="706870157cf5a49917b5c22fb771565954df7346"
CHECKOUT=Path("/kaggle/tmp/hala-h142-real-visible-heading-source")
OUTPUT=Path("/kaggle/working/hala-h142-private-visible-heading-lora")
RECEIPT=Path("/kaggle/working/hala-h142-private-train-attempt.json")
DATASET_SLUG="hala-h142-private-visible-heading-sft-v1"
ADAPTER_FOLDER="h140-private-original-v2-rank8-lora"

def record(payload):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=RECEIPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf8")
    tmp.replace(RECEIPT)

def checked(*args,timeout=180):
    r=subprocess.run(args,check=False,capture_output=True,text=True,
                     timeout=timeout)
    if r.returncode:
        raise RuntimeError("H142 reproducible PRIVATE free-T4 prerequisite failed")
    return r.stdout.strip()

def find_adapter(root):
    """Kaggle auto-extracts ZIP into a directory, preserving file hashes."""
    candidates=[p for p in root.rglob(ADAPTER_FOLDER) if p.is_dir()]
    if len(candidates)!=1:
        raise ValueError("H142 exactly one old 2B rank8 adapter source needed")
    adapter=candidates[0]
    names={x.name for x in adapter.iterdir() if x.is_file()}
    if not {"adapter_model.safetensors","adapter_config.json"}.issubset(names):
        raise ValueError("H142 old research adapter weights missing")
    return adapter

def main():
    status={
        "schema":"hala.h142.owner-private-t4-visible-heading-training-attempt.v1",
        "status":"H142_NO_MODEL_TRAINING_PERFORMED",
        "source_git_sha":SOURCE_SHA,
        "optimizer_steps":0,
        "required_optimizer_steps":64,
        "train_cases":64,"dev_cases":16,
        "h140_blind_neural_calls":0,
        "paid_compute_used":False,
        "flagship_Hala_35B_updated":False,
        "published_model_weights":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or CHECKOUT.exists() or OUTPUT.exists():
            raise ValueError("H142 requires fresh private Kaggle kernel")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H142 private FREE Tesla T4 only; no paid fallback")
        input_root=Path("/kaggle/input")
        if not input_root.is_dir():raise ValueError("H142 original owner source not mounted")
        # Avoid paths describing prior neural evaluations or holdout datasets.
        if any(any(word in p.name.lower() for word in ("blind","ood","h141"))
               for p in input_root.iterdir()):
            raise ValueError("H142 other original BLIND or neural evaluation mount forbidden")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=540)
        version={n:importlib.metadata.version(n).split("+")[0]
                 for n in ("torch","transformers","peft","Pillow")}
        if version!={"torch":"2.11.0","transformers":"5.17.0",
                     "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H142 exact pinned GPU packages drifted")
        CHECKOUT.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(CHECKOUT))
        checked("git","-C",str(CHECKOUT),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(CHECKOUT),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(CHECKOUT),"checkout","-q","--detach","FETCH_HEAD")
        if checked("git","-C",str(CHECKOUT),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H142 original training source version mismatch")
        sys.path.insert(0,str(CHECKOUT))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h142 import train_gpu
        from h140.qualification import digest
        train_gpu.DATA_ROOT=input_root
        sources=train_gpu.find_input_sources()
        adapter=find_adapter(input_root)
        if digest((adapter/"adapter_model.safetensors").read_bytes())!=train_gpu.PRETRAINED_V2_ADAPTER_SHA:
            raise ValueError("H142 owner-validated V2 adapter checkpoint missing")
        status.update(status="H142_RUNNING_GENUINE_FREE_T4_VISIBLE_HEADER_SFT",
                      input_source_train_sha=train_gpu.TRAIN_SHA,
                      input_source_dev_sha=train_gpu.DEV_SHA)
        record(status)
        evidence=train_gpu.run(OUTPUT,adapter)
        if (evidence.get("status")!="COMPLETED_64_REAL_FREE_T4_TRAIN_STEPS" or
            evidence.get("steps")!=64 or evidence.get("start_examples_seen")!=64 or
            evidence.get("visible_heading_examples_seen")!=128 or
            evidence.get("h140_blind_model_calls")!=0):
            raise ValueError("H142 actual original UI optimizer receipt incomplete")
        status.update(evidence,status="H142_COMPLETED_64_ORIGINAL_T4_OPTIMIZER_STEPS",
                      optimizer_steps=64)
        record(status)
    except Exception as exc:
        status.update(status="FAIL_H142_GPU_TRAINING_NOT_QUALIFIED",
                      failure_class=type(exc).__name__)
        record(status)
        raise

if __name__=="__main__":
    main()

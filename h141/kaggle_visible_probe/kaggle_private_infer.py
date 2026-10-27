"""Owner-private H141 original React screenshot visually-grounded model probe.

FREE T4 ONLY. Exactly 32 real external pretrained+V2 LoRA inference calls,
new fresh screenshot cohort. NO optimizer, NO previous H140 sealed BLIND.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="0ae8d253134c5b51fa9be2601f83fc89ff610b32"
REPO=Path("/kaggle/tmp/hala-h141-visible-research-source")
OUT=Path("/kaggle/working/hala-h141-visible-original-model-predictions")
RECEIPT=Path("/kaggle/working/hala-h141-private-visible-inference-attempt.json")
DATASET_SLUG="hala-h141-private-visible-probe-v1"
INPUT_FILENAME="model-visible-probe16-redacted-v1.zip"
ADAPTER_FILENAME="h140-private-original-v2-rank8-lora.zip"

def record(status):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    with RECEIPT.open("w",encoding="utf8") as fh:
        json.dump(status,fh,sort_keys=True,indent=2)
        fh.write("\n")

def checked(*command,timeout=180):
    r=subprocess.run(command,check=False,capture_output=True,text=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError("H141 free/private prerequisite command failed")
    return r.stdout.strip()

def identify_private_source(root):
    """No filename guesses from training or H140 blind mounts."""
    if not root.exists():
        raise ValueError("H141 owner source dataset missing")
    # Kaggle may rebase/private-mount datasets under arbitrary worker folders,
    # or add system-only directories. Frozen FILE content SHA is independently
    # verified by h141.gpu_visible_probe.input_admission/adapter_admission.
    children=[p for p in root.iterdir() if p.is_dir()]
    if any(any(word in p.name.lower() for word in ("blind","ood","train256","dev64"))
           for p in children):
        raise ValueError("H141 forbidden previously sealed training/BLIND mount")
    candidates={name:list(root.rglob(name)) for name in (INPUT_FILENAME,ADAPTER_FILENAME)}
    if any(len(files)!=1 for files in candidates.values()):
        print("H141_VISIBLE_PRIVATE_MOUNT_NAMES",
              sorted(p.name for p in children)[:15],flush=True)
        raise ValueError("H141 unique original private probe ZIP/adapter unavailable")
    if candidates[INPUT_FILENAME][0].parent!=candidates[ADAPTER_FILENAME][0].parent:
        raise ValueError("H141 split untrusted or unrelated model input mount")
    return candidates[INPUT_FILENAME][0],candidates[ADAPTER_FILENAME][0]

def main():
    status={"schema":"hala.h141.one-time-visible-original-inference-attempt.v1",
            "status":"NOT_STARTED",
            "source_git_sha":SOURCE_SHA,
            "model_inference_calls":0,
            "optimizer_steps":0,
            "trained_on_blind":False,
            "sealed_h140_blind_neural_calls":0,
            "paid_compute_used":False,
            "public_model_release":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists() or OUT.exists():
            raise ValueError("H141 private new Kaggle worker required")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H141 ONLY no-cost Nvidia Tesla T4")
        input_zip,adapter_zip=identify_private_source(Path("/kaggle/input"))
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        versions={name:importlib.metadata.version(name).split("+")[0]
                  for name in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                      "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H141 pinned free-T4 inference package versions drifted")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H141 pinned research source checkout incorrect")
        sys.path.insert(0,str(REPO))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h141.gpu_visible_probe import run,input_admission,adapter_admission
        input_admission(input_zip)
        # Do not extract or modify any H140 one-shot BLIND files.
        status["status"]="RUNNING_EXACT_32_REAL_VISIBLE_ORIGINAL_INFERENCES"
        record(status)
        result=run(input_zip,adapter_zip,OUT)
        if (result.get("status")!="H141_PRIVATE_VISIBLE_MODEL_2B_PREDICTIONS_NO_GOLD"
            or result.get("original_neural_calls")!=32 or
            result.get("gold_boxes_seen_by_model") is not False or
            result.get("sealed_h140_blind_calls")!=0 or
            result.get("real_optimizer_steps")!=0):
            raise ValueError("H141 original neural inference results cannot be certified")
        status.update(result,status="COMPLETED_32_REAL_VISIBLE_ORIGINAL_INFERENCES",
                      model_inference_calls=32)
        record(status)
    except Exception as exc:
        status.update(status="FAIL_H141_VISIBLE_NEURAL_INFERENCE_NOT_QUALIFIED",
                      failure_class=type(exc).__name__)
        record(status)
        raise

if __name__=="__main__":
    main()

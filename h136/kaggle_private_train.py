"""H136 Hala research-adapter: 64 real LoRA optimizer steps on FREE T4.

Only own original synthetic screenshots, private TRAIN288 and DEV24.
The SEALED TEST dataset is deliberately omitted from Kaggle metadata.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="fcff44ad3f252bf76a99e4bc26054787b3fd9825"
REPO=Path("/kaggle/tmp/hala-h136-source-and-training")
OUTPUT=Path("/kaggle/working/hala-h136-private-gui-start-end-lora")
RECEIPT=Path("/kaggle/working/hala-h136-private-training-attempt.json")

def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    temp.replace(RECEIPT)
def checked(*args,timeout=180):
    item=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if item.returncode:
        raise RuntimeError("H136 predeclared private training prerequisite failure")
    return item.stdout.strip()
def identify_owner_splits(inputs):
    """Admit only two genuine H136 source manifests regardless Kaggle mount layout."""
    SCHEMA="hala.h136.owner-original-workstation-gui-scenes.v1"
    found={"train":0,"dev":0}
    for file in Path(inputs).rglob("manifest.json"):
        try:
            item=json.loads(file.read_text("utf8"))
        except (OSError,ValueError,json.JSONDecodeError):
            continue
        if item.get("schema")!=SCHEMA:
            continue
        split=item.get("split")
        if split not in found:
            raise ValueError("H136 training kernel may not mount SEALED_TEST or unknown source split")
        found[split]+=1
    if found!={"train":1,"dev":1}:
        raise ValueError(f"H136 genuine private source mounts missing or duplicated: {found}")
    return found

def main():
    state={
        "schema":"hala.h136.private-2b-language-lora-training-attempt.v1",
        "status":"NOT_QUALIFIED",
        "source_git_sha":SOURCE_SHA,
        "optimizer_steps":0,
        "expected_optimizer_steps":64,
        "paid_compute_used":False,
        "owner_private_dataset":True,
        "sealed_holdout_attached":False,
        "external_dataset_training":False,
        "main_Hala_35B_weights_updated":False,
        "release_qualified":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H136 owner-only fresh Kaggle private noncommercial session")
        import torch
        if not torch.cuda.is_available() or not any(
                "T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("H136 only FREE Kaggle Nvidia T4; no paid fallback")
        inputs=Path("/kaggle/input")
        if not inputs.is_dir():
            raise ValueError("H136 Kaggle private datasets root absent")
        source_admission=identify_owner_splits(inputs)
        state["admitted_train_and_dev"]=source_admission
        state["stage"]="SOURCE_128_AND_16_ADMITTED"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        required={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        got={name:importlib.metadata.version(name).split("+")[0]
             for name in required}
        if got!=required:
            raise ValueError("H136 model-training package versions drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H136 immutable training code git SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
            HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["status"]="RUNNING_REAL_PRIVATE_H136_LANGUAGE_LORA"
        save(state)
        from h136.train_gpu import run
        result=run(OUTPUT)
        if (result.get("status")!="H136_REAL_64_OPTIMIZER_STEP_NATIVE_GUI_LORA_EXPERIMENT" or
            result.get("actual_optimizer_steps")!=64 or
            result.get("sealed_holdout_attached") is not False or
            result.get("main_Hala_flagship_weights_updated") is not False or
            result.get("paid_compute_used") is not False):
            raise ValueError("H136 real adapter training evidence incomplete")
        state.update(result)
        save(state)
    except Exception as exc:
        state.update(status="FAIL_H136_REAL_LORA_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        save(state)
        raise
if __name__=="__main__":
    main()

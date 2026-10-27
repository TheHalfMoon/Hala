"""H157 owner PRIVATE free Kaggle Tesla T4 second typed GUI LoRA research."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="838f2e9633fb5977b4404801fc7636658a92a6ee"
SOURCE_MANIFEST_SHA="a28b845b212ba7b745f1b155944174278b8aac69e69ad0f0eee3ca3acd8b7f78"
WORK=Path("/kaggle/tmp/hala-h157-source")
OUT=Path("/kaggle/working/hala-h157-spatial-action-lora")
RECEIPT=Path("/kaggle/working/hala-h157-private-t4-training-receipt.json")


def checked(*args,timeout=180):
    response=subprocess.run(args,capture_output=True,text=True,
                            check=False,timeout=timeout)
    if response.returncode:
        raise RuntimeError("H157 exact pinned free GPU prerequisite failed")
    return response.stdout.strip()


def store(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    temp.replace(RECEIPT)


def locate(root,basename,*,kind):
    root=Path(root)
    hits=list(root.rglob(basename))+list(root.rglob(basename+".zip"))
    hits=[p for p in hits if p.is_file() or p.is_dir()]
    if not hits and kind=="source":
        hits=[p.parent for p in root.rglob("manifest.json")
              if p.is_file() and
              json.loads(p.read_text("utf8")).get("schema")==
              "hala.h157.original-spatial-action-train-dev-v1"]
    if not hits and kind=="adapter":
        hits=[p.parent for p in root.rglob("adapter_model.safetensors")
              if p.is_file()]
    hits=list(dict.fromkeys(hits))
    if len(hits)!=1:
        raise ValueError("H157 exactly one original source and adapter required")
    return hits[0]


def admit_only_one_mount(root,src,weights):
    root=Path(root)
    if not root.is_dir():
        raise ValueError("Missing original owner private Kaggle mount")
    top=[p for p in root.iterdir() if p.is_dir()]
    if len(top)!=1:
        raise ValueError("H157 only one private dataset is allowed")
    mount=top[0].resolve()
    if (not src.resolve().is_relative_to(mount) or
        not weights.resolve().is_relative_to(mount) or
        src.resolve()==weights.resolve()):
        raise ValueError("H157 must use one owner private source dataset")


def main():
    receipt={
        "schema":"hala.h157.private-free-t4-spatial-action-sft.v1",
        "status":"H157_NO_MODEL_UPDATE_YET",
        "source_commit":SOURCE_COMMIT,
        "source_manifest_sha256":SOURCE_MANIFEST_SHA,
        "required_optimizer_steps":152,
        "optimizer_steps":0,"paid_gpu":False,
        "h140_sealed_blind_calls":0,
        "Hala_flagship_35B_updated":False,
        "model_weights_published":False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
            WORK.exists() or OUT.exists()):
            raise ValueError("H157 only NEW PRIVATE free Kaggle worker")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H157 no paid GPU fallback if free T4 unavailable")
        root=Path("/kaggle/input")
        if (not root.is_dir() or
            any(any(w in p.name.lower() for w in (
                "blind","h140","h153","h154","h155","h156-private-neural"))
                for p in root.iterdir())):
            raise ValueError("H157 H140/previous holdout mounts forbidden")
        source=locate(root,"h157-original-source",kind="source")
        adapter=locate(root,"h156-trained-typed-action-lora",kind="adapter")
        admit_only_one_mount(root,source,adapter)
        receipt.update(status="H157_PRIVATE_SOURCE_ADMITTED",
                       mount_names=[p.name for p in root.iterdir() if p.is_dir()])
        store(receipt)
        checked(sys.executable,"-m","pip","uninstall","-y",
                "torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2",
                "Pillow==11.3.0",timeout=540)
        actual={name:importlib.metadata.version(name).split("+")[0]
                for name in ("torch","transformers","peft","Pillow")}
        if actual!={"torch":"2.11.0","transformers":"5.17.0",
                    "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H157 source-pinned GPU package versions drifted")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(WORK))
        checked("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        checked("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H157 research source commit SHA not exact")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",
                          DO_NOT_TRACK="1",HALA_NONCOMMERCIAL_RESEARCH="1")
        from h157.train_gpu import run
        receipt["status"]="H157_RUNNING_REAL_152_ORIGINAL_SPATIAL_LORA_STEPS"
        store(receipt)
        result=run(source,adapter,OUT)
        if (result.get("status")!="H157_GENUINE_152_FREE_T4_OPTIMIZER_STEPS_COMPLETE"
            or result.get("optimizer_steps")!=152 or
            result.get("h140_sealed_blind_calls")!=0 or
            result.get("Hala35B_unchanged") is not True):
            raise ValueError("H157 GPU optimizer completion receipt invalid")
        receipt.update(
            status="H157_COMPLETED_152_REAL_FREE_T4_LORA_STEPS",
            optimizer_steps=152,
            trained_adapter_sha=result["new_trained_h157_adapter_sha256"],
            dev_before=result["before_dev"],
            dev_after=result["after_dev"])
        store(receipt)
    except Exception as error:
        receipt.update(status="FAIL_H157_NOT_TRAINED",
                       failure_class=type(error).__name__)
        store(receipt)
        raise


if __name__=="__main__":
    main()

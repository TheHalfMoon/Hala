"""H161 original two-app owner-private Kaggle FREE T4 mixed native LoRA trainer."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT="83c9d13b252c56badb5e0f3c2ef7b782cb79cd17"
WORK=Path("/kaggle/tmp/hala-h161-multi-app-sft-source")
OUT=Path("/kaggle/working/hala-h161-multi-original-app-trained-model")
RECEIPT=Path("/kaggle/working/h161-free-t4-training-receipt.json")


def command(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,
                          check=False,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H161 pinned original source setup failed")
    return result.stdout.strip()


def save(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    temp.replace(RECEIPT)


def find(root,basename):
    roots=list(Path(root).rglob(basename)) + list(
        Path(root).rglob(basename+".zip"))
    files=[p for p in dict.fromkeys(roots) if p.is_file() or p.is_dir()]
    if len(files)!=1:
        raise ValueError("H161 requires exactly one frozen original-app package")
    return files[0]


def main():
    doc={
        "schema":"hala.h161.private-real-two-app-genuine-SFT.v1",
        "source_commit":SOURCE_COMMIT,
        "status":"H161_NOT_TRAINED",
        "optimizer_steps":0,
        "paid_GPU":False,"h140_blind_calls":0,
        "DEV_gold_on_GPU":False,"Hala35B_unchanged":True,
        "model_weights_published":False,
    }
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
            WORK.exists() or OUT.exists()):
            raise ValueError("H161 owner-private new Kaggle T4 run only")
        import torch
        if (not torch.cuda.is_available() or
            not any("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))):
            raise ValueError("H161 never use paid GPU fallback")
        mount=Path("/kaggle/input")
        parents=[p for p in mount.iterdir() if p.is_dir()]
        if len(parents)!=1:
            raise ValueError("H161 one private Kaggle dataset only")
        names=("h161-train-original-app-source","h161-dev-original-app-source",
               "h157-frozen-research-lora")
        train,dev,adapter=[find(mount,name) for name in names]
        if (len({p.resolve() for p in (train,dev,adapter)})!=3 or
            any(not p.resolve().is_relative_to(parents[0].resolve())
                for p in (train,dev,adapter))):
            raise ValueError("H161 model source/data mounts not isolated")
        doc["status"]="H161_PRIVATE_TRAIN_DEV_MODEL_SOURCE_ADMITTED"
        save(doc)
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                timeout=540)
        versions={p:importlib.metadata.version(p).split("+")[0]
                  for p in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                     "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H161 pinned free T4 dependency versions unavailable")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H161 exact source SHA cannot be proven")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h161.train_gpu import run
        doc["status"]="H161_REAL_48_FREE_T4_OPTIMIZER_STEPS_EXECUTING"
        save(doc)
        result=run(train,dev,adapter,OUT)
        if (result.get("status")!=
            "H161_GENUINE_48_T4_TWO_REAL_APP_TRAIN_STEPS_COMPLETE" or
            result.get("optimizer_steps")!=48 or
            result.get("dev_labels_seen_by_GPU") is not False or
            result.get("h140_blind_calls")!=0):
            raise ValueError("H161 actual mixed-app research training incomplete")
        doc.update(status="H161_GENUINE_48_FREE_T4_OPTIMIZER_STEPS_COMPLETE",
                   optimizer_steps=48,
                   old_adapter_sha=result["starting_H157_adapter_sha256"],
                   new_adapter_sha=result["new_mixed_app_adapter_sha256"],
                   before_dev_format=result["before_dev_model_format"],
                   after_dev_format=result["after_dev_model_format"])
        save(doc)
    except Exception as exc:
        doc.update(status="H161_NO_QUALIFIED_TRAINED_MODEL",
                   failure_class=type(exc).__name__)
        save(doc)
        raise


if __name__=="__main__":
    main()

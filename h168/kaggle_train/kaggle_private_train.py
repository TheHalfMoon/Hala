"""H168 owner-private FREE Kaggle T4 real original browser TRAIN-only launcher."""
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_COMMIT="da7d20367b7bb1db439cf23fbff0700bfb0cab89"
DATA=Path("/kaggle/input")
PACKAGE_NAME="h168-training.pkg"
OUT=Path("/kaggle/working/hala-h168-private-real-original-train")
JOB=Path("/kaggle/working/h168-private-training-job.json")
WORK=Path("/kaggle/tmp/h168-source")

def call(*args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=600)
    if p.returncode:
        raise RuntimeError("H168 dependency or pinned source preparation failed")
    return p.stdout.strip()

def record(status,**extra):
    receipt=dict(schema="hala.h168.private-kaggle-train-only-job.v1",
         status=status,source_commit=SOURCE_COMMIT,
         paid_gpu=False,dev_ood_evaluations=0,
         prior_consumed_eval_used=False,model_release=False,
         flagship_35b_unchanged=True,**extra)
    JOB.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")

def main():
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or len(SOURCE_COMMIT)!=40:
            raise ValueError("H168 only pinned owner-private Kaggle job")
        files=[p for p in DATA.rglob("*") if p.is_file()]
        packages=[p for p in files if p.name==PACKAGE_NAME]
        language=[p for p in files if p.name=="adapter_model.safetensors"]
        if (len(files)!=4 or len(packages)!=1 or len(language)!=1
            or set(x.name for x in files)!={
                PACKAGE_NAME,"README.md","adapter_config.json","adapter_model.safetensors"}):
            raise ValueError("H168 Kaggle mounts contain unexpected files")
        source=language[0].parent
        if (len(set(p.parent for p in files))!=2 or
            "hala-h167-private-frozen-language-only-v1" not in source.parts or
            "hala-h168-private-train-32-v1" not in packages[0].parts):
            raise ValueError("H168 private research-only source mounts not qualified")
        import torch
        if not torch.cuda.is_available() or any(
                "T4" not in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("H168 require FREE T4, no paid fallback")
        call(sys.executable,"-m","pip","uninstall","-y","torchao")
        call(sys.executable,"-m","pip","install","-q",
             "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        call("git","init","-q",str(WORK))
        call("git","-C",str(WORK),"remote","add","origin",
             "https://github.com/TheHalfMoon/Hala.git")
        call("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        call("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if call("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H168 pinned GitHub source failed checkout")
        sys.path.insert(0,str(WORK))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        record("H168_REAL_ORIGINAL_FREE_T4_RUNNING",optimizer_steps=0)
        from h168.train_gpu import run
        report=run(packages[0],source,OUT)
        if (report["optimizer_steps"]!=32 or
            report["source_training_cases"]!=32 or
            report["dev_ood_evaluations"]!=0):
            raise ValueError("H168 32 original real app steps did not complete")
        record("H168_NEW_ORIGINAL_BROWSER_TRAIN_32_FREE_T4_PASS",
               optimizer_steps=32,
               peak_allocated_gib=report["peak_allocated_gib"],
               private_visual_sha256=report["saved_visual_only_adapter_sha256"])
    except Exception as exc:
        record("H168_REAL_ORIGINAL_TRAIN_NOT_QUALIFIED",
               optimizer_steps=0,
               failure_class=type(exc).__name__,
               failure_message=str(exc)[:200])
        raise

if __name__=="__main__":
    main()

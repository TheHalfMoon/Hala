"""Private H166 synthetic-only T4 visual-merger gradient/memory probe."""
import importlib.metadata
import json,os,subprocess,sys
from pathlib import Path

SOURCE_COMMIT="UNQUALIFIED"
WORK=Path("/kaggle/tmp/hala-h166-vision-probe-source")
OUT=Path("/kaggle/working/hala-h166-synthetic-vision-probe")
RECEIPT=Path("/kaggle/working/h166-job-receipt.json")

def command(*args,timeout=200):
    result=subprocess.run(args,capture_output=True,text=True,
                          check=False,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H166 pinned source/dependency preparation failed")
    return result.stdout.strip()

def record(doc):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    temp.replace(RECEIPT)

def main():
    receipt={"schema":"hala.h166.private-synthetic-visual-t4-job.v1",
       "status":"H166_NOT_PROBED","source_commit":SOURCE_COMMIT,
       "optimizer_steps":0,"real_app_training":False,
       "owner_gold_used":False,"new_ood_evaluations":0,
       "h140_blind_calls":0,"paid_GPU":False,
       "model_weights_published":False,"flagship_35B_unchanged":True}
    try:
        if (len(SOURCE_COMMIT)!=40 or WORK.exists() or OUT.exists() or
            not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
            raise ValueError("H166 exact source or Kaggle private run missing")
        import torch
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise ValueError("H166 requires a free Nvidia T4 CUDA GPU")
        if list(Path("/kaggle/input").iterdir()):
            raise ValueError("H166 synthetic-only probe must have NO owner dataset")
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q","transformers==5.17.0",
                "peft==0.21.2","Pillow==11.3.0",timeout=600)
        WORK.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_COMMIT:
            raise ValueError("H166 pinned GitHub code SHA changed")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        receipt["status"]="H166_SYNTHETIC_VISUAL_MERGER_STEP_RUNNING"
        record(receipt)
        from h166.vision_probe import run
        summary=run(OUT)
        if (summary["status"]!="H166_ONE_SYNTHETIC_VISUAL_MERGER_STEP_PASSED"
            or summary["optimizer_steps"]!=1 or
            summary["source_original_app_images"]!=0 or
            summary["model_weights_saved"] is not False):
            raise ValueError("H166 genuine one-step bounded research gate failed")
        receipt.update(status="H166_SYNTHETIC_FREE_T4_MEMORY_STEP_PASS",
                       optimizer_steps=1,
                       peak_allocated_gib=summary["peak_allocated_gib"],
                       trainable_parameters=summary["trainable_parameters"])
        record(receipt)
    except Exception as exc:
        receipt.update(status="H166_SYNTHETIC_T4_PROBE_UNQUALIFIED",
                       failure_class=type(exc).__name__)
        record(receipt)
        raise

if __name__=="__main__":
    main()

"""H167 Kaggle-private combined-adapter synthetic-only feasibility launcher."""
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_COMMIT="2d5637e38cb49ae4e4b23796e806c0b363a4d0e1"
PACKAGE=Path("/kaggle/input/hala-h167-private-frozen-language-only-v1")
OUTPUT=Path("/kaggle/working/hala-h167-private-combined-t4")
JOB=Path("/kaggle/working/h167-combined-job-receipt.json")
WORK=Path("/kaggle/tmp/h167-frozen-source")

def command(*args):
    subprocess.run(args,check=True,capture_output=True,timeout=600)

def record(state,**details):
    doc=dict(schema="hala.h167.private-combined-language-visual-kaggle-job.v1",
             status=state,source_commit=SOURCE_COMMIT,
             owner_gold_used=False,previous_ood_or_dev_used=False,
             new_ood_evaluations=0,paid_gpu=False,
             model_weights_published=False,flagship_35b_unchanged=True,**details)
    JOB.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n",encoding="utf8")

def main():
    try:
        if (len(SOURCE_COMMIT)!=40 or
            len(list(Path("/kaggle/input").iterdir()))!=1 or
            not PACKAGE.is_dir() or
            {x.name for x in PACKAGE.iterdir()}!={"README.md","adapter_config.json","adapter_model.safetensors"} or
            not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
            raise ValueError("H167 private Kaggle and frozen single-source ZIP required")
        command(sys.executable,"-m","pip","uninstall","-y","torchao")
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        WORK.parent.mkdir(exist_ok=True,parents=True)
        command("git","init","-q",str(WORK))
        command("git","-C",str(WORK),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_COMMIT)
        command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(["git","-C",str(WORK),"rev-parse","HEAD"],
                text=True).strip()!=SOURCE_COMMIT:
            raise ValueError("H167 source SHA mismatch")
        sys.path.insert(0,str(WORK))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        record("H167_SYNTHETIC_COMBINED_RUNNING",optimizer_steps=0)
        from h167.combined_probe import run
        result=run(PACKAGE,OUTPUT)
        record("H167_SYNTHETIC_COMBINED_FREE_T4_PASS",
               optimizer_steps=result["optimizer_steps"],
               peak_allocated_gib=result["peak_allocated_gib"],
               trainable_visual_parameters=result["trainable_visual_parameters"])
    except Exception as e:
        record("H167_SYNTHETIC_COMBINED_T4_UNQUALIFIED",
               optimizer_steps=0,failure_class=type(e).__name__)
        raise

if __name__=="__main__":
    main()

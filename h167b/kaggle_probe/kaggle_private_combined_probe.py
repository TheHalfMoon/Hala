"""H167b isolated private Kaggle T4 PEFT dual-adapter regression run."""
import json
import os
import subprocess
import sys
from pathlib import Path

SHA="ae69beef93521d060a2d8a9f5620785b21f55c1f"
OUTPUT=Path("/kaggle/working/hala-h167-private-combined-t4")
JOB=Path("/kaggle/working/h167-combined-job-receipt.json")
WORK=Path("/kaggle/tmp/h167b-pinned-code")

def call(*args):
    subprocess.run(args,check=True,capture_output=True,timeout=600)

def receipt(status,**details):
    data=dict(schema="hala.h167.private-combined-language-visual-kaggle-job.v1",
              status=status,source_commit=SHA,optimizer_steps=0,
              owner_gold_used=False,previous_ood_or_dev_used=False,
              new_ood_evaluations=0,paid_gpu=False,
              model_weights_published=False,flagship_35b_unchanged=True)
    data.update(details)
    JOB.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf8")

def main():
    try:
        matches=list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
        if len(matches)!=1 or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H167b needs exact private frozen research adapter only")
        source=matches[0].parent
        if (set(p.name for p in source.iterdir())
                !={"README.md","adapter_config.json","adapter_model.safetensors"} or
            "hala-h167-private-frozen-language-only-v1" not in source.parts or
            len([p for p in Path("/kaggle/input").rglob("*") if p.is_file()])!=3):
            raise ValueError("H167b unexpected data or private adapter layout")
        call(sys.executable,"-m","pip","uninstall","-y","torchao")
        call(sys.executable,"-m","pip","install","-q",
             "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        call("git","init","-q",str(WORK))
        call("git","-C",str(WORK),"remote","add","origin",
             "https://github.com/TheHalfMoon/Hala.git")
        call("git","-C",str(WORK),"fetch","--depth","1","origin",SHA)
        call("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(["git","-C",str(WORK),"rev-parse","HEAD"],
                text=True).strip()!=SHA:
            raise ValueError("H167b source commit mismatch")
        sys.path.insert(0,str(WORK))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        receipt("H167B_COMBINED_SYNTHETIC_RUNNING")
        from h167.combined_probe import run
        result=run(source,OUTPUT)
        if result["optimizer_steps"]!=2:
            raise ValueError("H167b exactly two synthetic steps required")
        receipt("H167_SYNTHETIC_COMBINED_FREE_T4_PASS",
                optimizer_steps=2,
                peak_allocated_gib=result["peak_allocated_gib"],
                trainable_visual_parameters=result["trainable_visual_parameters"])
    except Exception as exc:
        receipt("H167_SYNTHETIC_COMBINED_T4_UNQUALIFIED",
                failure_class=type(exc).__name__,
                failure_message=str(exc)[:200])
        raise

if __name__=="__main__":
    main()

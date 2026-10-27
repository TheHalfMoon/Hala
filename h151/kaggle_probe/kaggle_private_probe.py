"""H151 private free-T4 independent shifted-original Qwen2B inference.

No fitting/optimizer, no oracle model inputs and NO H140 previously sealed
BLIND; one SHA-frozen original shifted-layout cohort and H142 adapter only.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="76f3e90b76d5de4d8c78d1890f818154d9ac0b9f"
WORK=Path("/kaggle/tmp/hala-h151-frozen-research-source")
OUT=Path("/kaggle/working/hala-h151-independent-shifted-original-inference")
RECEIPT=Path("/kaggle/working/hala-h151-shifted-inference-receipt.json")
IMAGES="model-shifted16-redacted"
ADAPTER="h142-private-visible-column-lora"

def run_command(*args,timeout=210):
    r=subprocess.run(args,capture_output=True,text=True,check=False,timeout=timeout)
    if r.returncode:
        raise RuntimeError("H151 pinned free/private inference prerequisite failed")
    return r.stdout.strip()
def write(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    RECEIPT.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf8")
def find_source(root,name):
    candidates=list(root.rglob(name))+list(root.rglob(name+".zip"))
    if len(candidates)!=1:
        raise ValueError("H151 unique owner SHA-pinned source path missing: "+name)
    return candidates[0]
def main():
    status={"schema":"hala.h151.private-one-time-shifted-original-inference-receipt.v1",
            "status":"NOT_STARTED","source_git_sha":SOURCE_SHA,
            "model_calls":0,"optimizer_steps":0,
            "paid_compute_used":False,"h140_blind_calls":0,
            "main_hala_35B_updated":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or WORK.exists() or OUT.exists():
            raise ValueError("H151 requires fresh Kaggle worker")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ):
            raise ValueError("H151 FREE owner-private Tesla T4 only")
        root=Path("/kaggle/input")
        children=[x.name.lower() for x in root.iterdir()]
        if any(any(marker in n for marker in
                   ("blind","ood","train64","dev16","h140"))
               for n in children):
            raise ValueError("H151 inference cannot mount old training/BLIND sources")
        images=find_source(root,IMAGES)
        adapter=find_source(root,ADAPTER)
        if images.parent!=adapter.parent:
            raise ValueError("H151 model and private images must share source dataset root")
        run_command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        run_command(sys.executable,"-m","pip","install","-q",
                    "transformers==5.17.0","peft==0.21.2","Pillow==11.3.0",
                    timeout=540)
        versions={n:importlib.metadata.version(n).split("+")[0]
                  for n in ("torch","transformers","peft","Pillow")}
        if versions!={"torch":"2.11.0","transformers":"5.17.0",
                      "peft":"0.21.2","Pillow":"11.3.0"}:
            raise ValueError("H151 pinned GPU package versions mismatched")
        WORK.parent.mkdir(parents=True,exist_ok=True)
        run_command("git","init","-q",str(WORK))
        run_command("git","-C",str(WORK),"remote","add","origin",
                    "https://github.com/TheHalfMoon/Hala.git")
        run_command("git","-C",str(WORK),"fetch","--depth","1","origin",SOURCE_SHA)
        run_command("git","-C",str(WORK),"checkout","--detach","-q","FETCH_HEAD")
        if run_command("git","-C",str(WORK),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H151 original research source SHA drift")
        sys.path.insert(0,str(WORK))
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",
                          HALA_NONCOMMERCIAL_RESEARCH="1")
        from h151.gpu_probe import admit_images,run,ADAPTER_MODEL_SHA
        admit_images(images)
        status["status"]="RUNNING_NEW_SHIFTED_GUI_32_REAL_NEURAL_INFERENCES"
        write(status)
        result=run(images,adapter,OUT)
        if (result.get("real_gpu_inference_calls")!=32 or
            result.get("gold_seen") is not False or
            result.get("h140_blind_calls")!=0):
            raise ValueError("H151 model outputs incomplete or contaminated")
        status.update(status="COMPLETE_NEW_SHIFTED_GUI_32_REAL_NEURAL_CALLS",
                      model_calls=32,
                      adapter_sha256=ADAPTER_MODEL_SHA)
        write(status)
    except Exception as exc:
        status.update(status="FAIL_NEURAL_EVAL_NOT_QUALIFIED",
                      failure_class=type(exc).__name__)
        write(status)
        raise
if __name__=="__main__":
    main()

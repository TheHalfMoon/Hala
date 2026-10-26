"""H135 frozen 2x2 image preprocessing pilot on private FREE Kaggle T4."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="__GIT_SHA__"
REPO=Path("/kaggle/tmp/hala-h135-frozen-resolution-source")
OUTPUT=Path("/kaggle/working/hala-h135-private-native-versus-640-pilot12")
RECEIPT=Path("/kaggle/working/hala-h135-pilot-attempt.json")
def save(item):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    temp=RECEIPT.with_suffix(".tmp")
    temp.write_text(json.dumps(item,sort_keys=True,indent=2)+"\n","utf8")
    temp.replace(RECEIPT)
def check_command(*args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("H135 exact private frozen inference prerequisites unavailable")
    return p.stdout.strip()

def run():
    receipt={
      "schema":"hala.h135.private-pretrained-resolution-pilot-attempt.v1",
      "status":"NOT_QUALIFIED",
      "git_source_sha":SOURCE_SHA,"paid_compute":False,"optimizer_steps":0,
      "trained_adapter_unchanged":True,"no_sealed_images_attached":True,
      "release_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H135 new owner-private Kaggle GPU kernel only")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H135 PRIVATE FREE T4 only; no paid GPU fallback")
        check_command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        check_command(sys.executable,"-m","pip","install","-q",
                      "transformers==5.17.0","peft==0.21.2",
                      "Pillow==11.3.0",timeout=540)
        expected={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        observed={k:importlib.metadata.version(k).split("+")[0] for k in expected}
        if observed!=expected:
            raise ValueError("H135 external research model runtime version drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        check_command("git","init","-q",str(REPO))
        check_command("git","-C",str(REPO),"remote","add","origin",
                      "https://github.com/TheHalfMoon/Hala.git")
        check_command("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        check_command("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if check_command("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H135 committed frozen pilot code revision mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        receipt["stage"]="FROZEN_PRETRAINED_2x2_TRUE_NEURAL_INFERENCES"
        save(receipt)
        from h135.ablate_gpu import execute
        output=execute(OUTPUT)
        if (output.get("status")!="H135_REAL_PRIVATE_PILOT12_RESOLUTION_PAIRED" or
            output.get("real_neural_calls")!=96 or
            output.get("actual_optimizer_steps")!=0 or
            output.get("paid_cloud") is not False):
            raise ValueError("H135 real frozen 96 neural calls incomplete")
        receipt.update(output);save(receipt)
    except Exception as exc:
        receipt.update(status="FAIL_H135_PILOT_REAL_NEURAL_NOT_QUALIFIED",
                       failure_class=type(exc).__name__)
        save(receipt)
        raise
if __name__=="__main__": run()

"""Owner-private Kaggle H97 vision-only 32 new real Qwen optimizer steps."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="1b17d46d3f722f1a48e7c3b9a411bd0806b17db9"
REPO=Path("/kaggle/tmp/hala-h97-source")
SCRATCH=Path("/kaggle/tmp/hala-h97-study")
OUTPUT=Path("/kaggle/working/hala-h97-vision-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h97-attempt.json")

def record(value):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2),"utf8")
    temp.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,timeout=timeout,
                   capture_output=True,text=True)

def main():
    data={"schema":"hala.h97.private-vision-32-step-attempt.v1",
          "source_sha":SOURCE_SHA,"status":"STARTED","stage":"GPU_PREFLIGHT",
          "verified_optimizer_steps":0,"public_model_released":False,
          "commercial_use":False,"host_readback_verified":False}
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or not Path("/kaggle/working").is_dir()):
            raise ValueError("actual private Kaggle notebook only")
        import torch
        if (not torch.cuda.is_available()
                or not any("T4" in torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count()))):
            raise ValueError("actual free Tesla T4 GPU required")
        data["gpu_names"]=[torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
        data["stage"]="PINNED_GPU_PACKAGES"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=480)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        found={k:importlib.metadata.version(k) for k in expected}
        if any(found[k].split("+")[0]!=v for k,v in expected.items()):
            raise ValueError("original pinned Qwen dependencies changed")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible optional torchao installed")
        data["stage"]="IMMUTABLE_PUBLIC_SOURCE"
        if REPO.exists():
            raise ValueError("existing unqualified Git source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",
                SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if subprocess.check_output(
                ["git","-C",str(REPO),"rev-parse","HEAD"],
                text=True).strip()!=SOURCE_SHA:
            raise ValueError("source commit differs")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        sys.path.insert(0,str(REPO))
        data["stage"]="REAL_32_VISUAL_OPTIMIZER_STEPS"
        record(data)
        from h97.research_train import run
        proof=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (proof.get("status")!="REAL_32_VISION_LORA_CHECKPOINT_WRITTEN"
                or proof.get("new_optimizer_steps")!=32
                or proof.get("gradient_bearing_steps")!=32
                or not proof.get("checkpoint_sha256")):
            raise ValueError("32 real visual checkpoint steps not confirmed")
        data.update(status="REAL_32_VISION_LORA_CHECKPOINT_WRITTEN",
                    stage="COMPLETE_PRIVATE_OPTIMIZER_CHECKPOINT",
                    verified_optimizer_steps=32,
                    checkpoint_sha256=proof["checkpoint_sha256"])
        record(data)
        print("HALA_H97_32_REAL_VISUAL_UPDATES_RECORDED",flush=True)
    except Exception as error:
        data.update(status="FAILED_NO_QUALIFIED_32_STEP_CHECKPOINT",
                    failure_class=type(error).__name__)
        record(data)
        print("HALA_H97_STUDY_FAILED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

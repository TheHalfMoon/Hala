"""H99 Kaggle PRIVATE original-H71 balanced 128-step visual research executor."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="b4ad37c04a9b3fa9311f0c4b15fb277b50f5d570"
REPO=Path("/kaggle/tmp/hala-h99-source")
SCRATCH=Path("/kaggle/tmp/hala-h99-balanced-source")
OUTPUT=Path("/kaggle/working/hala-h99-balanced-visual-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h99-attempt.json")

def record(value):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,text=True,
                   capture_output=True,timeout=timeout)

def main():
    data={"schema":"hala.h99.private-original-h71-balanced-attempt.v1",
          "status":"STARTED","stage":"GPU_PREFLIGHT",
          "pinned_source_sha":SOURCE_SHA,"optimizer_steps_verified":0,
          "public_model_released":False,"commercial_use":False,
          "independent_host_adapter_sha_readback":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private Kaggle runtime required")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("genuine free Kaggle Tesla T4 GPU required")
        data["gpu_names"]=[torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
        data["stage"]="PINNED_FREE_GPU_MODEL_DEPENDENCIES"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=480)
        requirements={"torch":"2.11.0","transformers":"5.16.1",
                      "peft":"0.20.0","Pillow":"11.3.0"}
        present={k:importlib.metadata.version(k) for k in requirements}
        if any(present[k].split("+")[0]!=v for k,v in requirements.items()):
            raise ValueError("Qwen pinned training library drift")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible torchao present")
        data["stage"]="EXACT_GIT_SOURCE"
        if REPO.exists():
            raise ValueError("preexisting unqualified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=30)
        current=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if current!=SOURCE_SHA:
            raise ValueError("immutable sampler/trainer source SHA mismatch")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",
                          DO_NOT_TRACK="1")
        sys.path.insert(0,str(REPO))
        data["stage"]="REAL_ORIGINAL_10K_H71_SAMPLER_AND_GPU_UPDATES"
        record(data)
        from h99.research_train import run
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result.get("status")!="REAL_BALANCED_128_VISUAL_UPDATES_RECORDED"
                or result.get("new_optimizer_steps")!=128
                or result.get("positive_gradient_steps")!=128
                or result.get("actual_unique_optimizer_records")!=128
                or not result.get("checkpoint_sha256")):
            raise ValueError("128 real balanced visual updates not confirmed")
        data.update(status="REAL_128_BALANCED_VISUAL_OPTIMIZER_UPDATES",
                    stage="CHECKPOINT_PERSISTED_PRIVATE",
                    optimizer_steps_verified=128,
                    checkpoint_sha256=result["checkpoint_sha256"],
                    plan_sha256=result["sampler_plan_sha256"])
        record(data)
        print("H99_128_REAL_BALANCED_VISION_OPTIMIZER_STEPS_VERIFIED",flush=True)
    except Exception as error:
        data.update(status="FAILED_NO_QUALIFIED_BALANCED_TRAINING",
                    failure_class=type(error).__name__)
        record(data)
        print("H99_RESEARCH_FAILED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

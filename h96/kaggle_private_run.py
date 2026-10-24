"""H96 one-step vision-only LoRA: private Kaggle T4 execution entrypoint."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="595658253040d4802d1b97babf868a66e2222781"
REPO=Path("/kaggle/tmp/hala-h96-source")
SCRATCH=Path("/kaggle/tmp/hala-h96-vision-pilot")
OUTPUT=Path("/kaggle/working/hala-h96-vision-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h96-attempt.json")


def receipt(payload):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,timeout=timeout,
                   capture_output=True,text=True)


def main():
    state={"schema":"hala.h96.private-real-vision-lora-attempt.v1",
           "status":"STARTED","stage":"GPU_PREFLIGHT",
           "source_sha":SOURCE_SHA,"steps_verified":0,
           "commercial_use":False,"public_model_released":False,
           "downloaded_checkpoint_verified":False}
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or not Path("/kaggle/working").is_dir()):
            raise ValueError("genuine private Kaggle Linux notebook required")
        import torch
        if (not torch.cuda.is_available() or
                not any("T4" in torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count()))):
            raise ValueError("genuine free Kaggle Tesla T4 required")
        state["gpu_names"]=[torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
        state["stage"]="PINNED_DEPENDENCIES"
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=480)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        found={key:importlib.metadata.version(key) for key in expected}
        if any(found[k].split("+")[0]!=v for k,v in expected.items()):
            raise ValueError("Qwen pinned vision training dependencies mismatched")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("old optional torchao conflicts with PEFT")
        state["stage"]="PINNED_GITHUB_SOURCE"
        if REPO.exists():
            raise ValueError("prior unverified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",
                SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("source SHA mismatch")
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        sys.path.insert(0,str(REPO))
        state["stage"]="REAL_VISION_ONLY_ONE_STEP"
        receipt(state)
        from h96.research_probe import run
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result.get("status")!="REAL_ONE_STEP_VISION_LORA_CHECKPOINT_WRITTEN"
                or result.get("optimizer_steps")!=1
                or not result.get("checkpoint_sha256")):
            raise ValueError("no real vision LoRA checkpoint from T4")
        state.update(status="REAL_VISION_ONE_STEP_CHECKPOINT_WRITTEN",
                     stage="PRIVATE_CHECKPOINT_WRITTEN",steps_verified=1,
                     checkpoint_sha256=result["checkpoint_sha256"])
        receipt(state)
        print("HALA_H96_ONE_REAL_VISION_OPTIMIZER_STEP_WRITTEN",flush=True)
    except Exception as error:
        state.update(status="FAILED_NO_VERIFIED_VISUAL_CHECKPOINT",
                     failure_class=type(error).__name__)
        receipt(state)
        print("HALA_H96_VISION_PROBE_FAILED",type(error).__name__,flush=True)
        raise


if __name__=="__main__":
    main()

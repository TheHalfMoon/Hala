"""H92 Kaggle PRIVATE research kernel: 8 new warm-started Qwen LoRA updates.

No public model weights, no paid compute, no publication. Exact original H87
private Kaggle kernel must be attached as kernel_source; original H71 as dataset.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PINNED_GIT_SHA="81a8e3f191f472c1b35059cadccb32e0dc46d00c"
SOURCE_URL="https://github.com/TheHalfMoon/Hala.git"
REPO=Path("/kaggle/tmp/hala-h92-source")
WORK=Path("/kaggle/tmp/hala-h92-study")
OUTPUT=Path("/kaggle/working/hala-h92-private-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h92-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding="utf8")
    temp.replace(ATTEMPT)

def command(*args,timeout=120):
    subprocess.run(list(args),check=True,timeout=timeout,capture_output=True,text=True)

def run():
    state={"schema":"hala.h92.private-kaggle-warmstart-attempt.v1",
           "source_sha":PINNED_GIT_SHA,"status":"STARTED",
           "stage":"KAGGLE_RESEARCH_PREFLIGHT",
           "verified_optimizer_steps":0,
           "public_model_released":False,
           "commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("real Kaggle private Linux notebook required")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<1:
            raise RuntimeError("real CUDA GPU required")
        state["gpu_names"]=[torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
        state["stage"]="PINNED_DEPENDENCIES"
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=480)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        actual={name:importlib.metadata.version(name) for name in expected}
        if any(actual[n].split("+")[0]!=v for n,v in expected.items()):
            raise ValueError("pinned model runtime mismatch")
        try:
            importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise ValueError("incompatible optional torchao still installed")
        state["stage"]="EXACT_H92_GIT_SOURCE"
        if REPO.exists():
            raise ValueError("refusing unverified preexisting Git checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(REPO),timeout=30)
        command("git","-C",str(REPO),"remote","add","origin",SOURCE_URL)
        command("git","-C",str(REPO),"fetch","--depth","1","origin",PINNED_GIT_SHA,timeout=150)
        command("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=30)
        sha=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],
                                    text=True,timeout=15).strip()
        if sha!=PINNED_GIT_SHA:
            raise ValueError("training source SHA mismatch")
        os.environ.update(
            HALA_NONCOMMERCIAL_RESEARCH="1",
            HALA_REMOTE_TRAINING_ALLOWED="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1"
        )
        sys.path.insert(0,str(REPO))
        state["stage"]="EIGHT_REAL_QWEN_LORA_UPDATES"
        record(state)
        from h92.research_train import run as train_research
        result=train_research(repo=REPO,scratch=WORK,output=OUTPUT)
        if (result.get("status")!="EIGHT_NEW_QWEN_LORA_OPTIMIZER_STEPS_VERIFIED"
                or result.get("new_optimizer_steps")!=8
                or not result.get("checkpoint_sha256")):
            raise ValueError("no eight-step complete optimizer-bearing checkpoint")
        state.update(stage="COMPLETE_CHECKPOINT_WRITTEN",
                     status="EIGHT_NEW_QWEN_LORA_OPTIMIZER_STEPS_VERIFIED",
                     verified_optimizer_steps=8,
                     checkpoint_sha256=result["checkpoint_sha256"],
                     seed_adapter_sha256=result["h87_init_adapter_sha256"])
        record(state)
        print("HALA_H92_REAL_MODEL_EIGHT_STEPS_COMPLETE",flush=True)
    except Exception as exc:
        state.update(status="FAILED_NO_NEW_EIGHT_STEP_QUALIFICATION",
                     stage=state["stage"],failure_class=type(exc).__name__)
        record(state)
        print("HALA_H92_RESEARCH_BLOCKED",type(exc).__name__,flush=True)
        raise

if __name__=="__main__":
    run()

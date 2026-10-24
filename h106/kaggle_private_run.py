"""H106 private Kaggle FREE T4 joint source-only research launcher.

Never uploads model weights, original source IDs/images or checkpoints to HF.
No paid providers. Results are private and require independent host readback
and strictly fresh GUI evaluation before any model improvement claim.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="55fc8e78ea273247b84241985aad863c41673454"
REPO=Path("/kaggle/tmp/hala-h106-source")
SCRATCH=Path("/kaggle/tmp/hala-h106-original-H71-joint")
OUTPUT=Path("/kaggle/working/hala-h106-private-joint-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h106-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    staging=ATTEMPT.with_suffix(".tmp")
    staging.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    staging.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,check=True,timeout=timeout)


def main():
    status={"schema":"hala.h106.private-real-joint-lora-attempt.v1",
            "status":"STARTED","stage":"T4_FREE_NONCOMMERCIAL_PREFLIGHT",
            "exact_source_sha":SOURCE_SHA,
            "completed_joint_optimizer_steps_verified":0,
            "checkpoint_host_SHA_verified":False,
            "model_GUI_quality_qualified":False,
            "weights_published":False,"commercial_use":False,"paid_resources_used":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("actual owner-private Kaggle notebook required")
        import torch
        if not torch.cuda.is_available() or not any(
             "T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
            raise ValueError("real Kaggle free T4 accelerator required")
        status["gpu_names"]=[torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())]
        status["stage"]="PINNED_NONCOMMERCIAL_MODEL_DEPENDENCIES"
        record(status)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0",
                "Pillow==11.3.0",timeout=540)
        required={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        present={k:importlib.metadata.version(k).split("+")[0] for k in required}
        if present!=required:
            raise ValueError("H106 pinned original training runtime changed")
        status["stage"]="IMMUTABLE_GIT_JOINT_TRAINER_SOURCE"
        record(status)
        if REPO.exists():raise ValueError("existing unverified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        sha=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if sha!=SOURCE_SHA:
            raise ValueError("H106 trained source SHA no longer pinned")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        status["stage"]="REAL_256_FRESH_JOINT_VISION_LANGUAGE_LORA_OPTIMIZER"
        record(status)
        from h106.research_train import run
        output=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (output["status"]!="ORIGINAL_H71_REAL_256_JOINT_OPTIMIZER_STEPS_VERIFIED"
                or output["new_optimizer_steps"]!=256
                or output["gradient_bearing_steps"]!=256
                or output["verified_unique_original_train_screenshots"]!=256):
            raise ValueError("H106 actual original joint optimizer updates unqualified")
        status.update(status=output["status"],
                      stage="PRIVATE_CHECKPOINT_RECEIPT",
                      completed_joint_optimizer_steps_verified=256,
                      checkpoint_sha256=output["checkpoint_sha256"],
                      adapter_sha256=output["adapter_sha256"])
        record(status)
        print("H106_REAL_256_JOINT_TRAIN_UPDATES_CHECKPOINT_PRIVATE",flush=True)
    except Exception as error:
        status["status"]="FAILED_H106_QUALIFIED_JOINT_TRAINING"
        status["failure_class"]=type(error).__name__
        record(status)
        print("H106_JOINT_SOURCE_TRAINING_BLOCKED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":main()

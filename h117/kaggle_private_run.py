"""H117 private owner free T4 signed destination pilot. New fresh LoRA only."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="c89f5ebd867e569e093eed2ad602d5342b7a19b7"
REPO=Path("/kaggle/tmp/hala-h117-frozen-source")
SCRATCH=Path("/kaggle/tmp/hala-h117-original-h113-conditioned-delta")
OUTPUT=Path("/kaggle/working/hala-h117-private-conditioned-delta256-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h117-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    staging=ATTEMPT.with_suffix(".tmp")
    staging.write_text(json.dumps(state,indent=2,sort_keys=True),"utf8")
    staging.replace(ATTEMPT)

def checked(*cmd,timeout=120):
    subprocess.run(list(cmd),check=True,capture_output=True,text=True,timeout=timeout)

def main():
    state={
      "schema":"hala.h117.private-start-conditioned-signed-END-T4.v1",
      "status":"STARTED","stage":"PRIVATE_NONCOMMERCIAL_FREE_T4_ADMISSION",
      "frozen_code_sha":SOURCE_SHA,
      "actual_verified_optimizer_steps":0,
      "model_gui_skill_qualified":False,
      "model_weights_published":False,"paid_resources":False,
      "commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner private Kaggle research only")
        import torch
        if not torch.cuda.is_available() or not any(
          "T4" in torch.cuda.get_device_name(i)
          for i in range(torch.cuda.device_count())):
            raise ValueError("free research Kaggle T4 not assigned")
        state["stage"]="EXACT_ORIGINAL_QWEN3VL_TRAINING_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        versions={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0]
            for k in versions}!=versions:
            raise ValueError("Pinned Qwen vision stack version changed")
        state["stage"]="ORIGINAL_SOURCE_SHA_BOUND_PRIVATE_CHECKOUT"
        record(state)
        if REPO.exists():raise ValueError("unverified prior source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:raise ValueError("H117 exact source SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ORIGINAL_TRAIN_ONLY_256_ACTUAL_START_SIGNED_END_OPTIMIZER"
        record(state)
        from h117.research_train import run
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result["status"]!="H117_REAL_256_ORIGINAL_START_CONDITIONED_DELTA_UPDATES_VERIFIED"
                or result["actual_optimizer_steps"]!=256
                or result["positive_gradient_and_changed_weights_steps"]!=256
                or result["pilot_source_images"]!=128
                or result["start_point_training_rows"]!=128
                or result["conditioned_delta_training_rows"]!=128
                or result["original_h116_conditioned_signed_delta_sha256"]!=
                   "174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c"):
            raise ValueError("actual signed destination optimizer qualification incomplete")
        state.update(
            status=result["status"],
            stage="OWNER_PRIVATE_SIGNED_DESTINATION_ADAPTER_RECEIPT",
            actual_verified_optimizer_steps=256,
            adapter_sha256=result["adapter_sha256"],
            checkpoint_sha256=result["checkpoint_sha256"],
            original_h116_signed_delta_sha256=result["original_h116_conditioned_signed_delta_sha256"])
        record(state)
        print("H117_REAL_ORIGINAL_256_CONDITIONED_DELTA_GPU_STEPS_VERIFIED",flush=True)
    except Exception as err:
        state.update(status="FAILED_H117_MODEL_TRAINING_NOT_QUALIFIED",
                     failure_class=type(err).__name__)
        record(state)
        print("H117_TRAIN_NOT_QUALIFIED",type(err).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

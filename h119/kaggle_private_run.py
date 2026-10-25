"""H119 private owner noncommercial FREE T4 original full512 TRAIN experiment."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="688cafb298c66840d38c192b2bc56ffed7098601"
REPO=Path("/kaggle/tmp/hala-h119-frozen-source")
SCRATCH=Path("/kaggle/tmp/hala-h119-original-train-512-anchor-robust")
OUTPUT=Path("/kaggle/working/hala-h119-private-robust512-1024-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h119-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,indent=2,sort_keys=True),"utf8")
    tmp.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,
                   check=True,timeout=timeout)


def main():
    state={
        "schema":"hala.h119.private-real-full512-1024-noncommercial-T4.v1",
        "status":"STARTED","stage":"PRIVATE_OWNER_FREE_T4_CHECK",
        "frozen_code_sha":SOURCE_SHA,"actual_verified_optimizer_steps":0,
        "paid_resources_used":False,"commercial_use":False,
        "model_weights_published":False,"general_GUI_skill_qualified":False,
        "hidden_TEST_used":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("Private original TRAIN only on Kaggle free T4")
        import torch
        if not torch.cuda.is_available() or not any(
                "T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("No eligible free Kaggle T4 accelerator")
        state["stage"]="EXACT_ORIGINAL_QWEN3VL_RUNTIME"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0",
                "Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("Pinned Qwen vision stack drift")
        state["stage"]="SHA_PINNED_NEW_TRAINER_SOURCE"
        record(state)
        if REPO.exists():
            raise ValueError("Preexisting unverified training checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1",
                "origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("Actual training source SHA differs")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="1024_REAL_OPTIMIZER_STEPS_ORIGINAL_TRAIN_ONLY"
        record(state)
        from h119.research_train import run,EXPECTED_TRAIN_SHA
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result["status"]!="H119_REAL_1024_ORIGINAL_FULL512_ROBUST_TRAIN_STEPS_VERIFIED"
                or result["actual_optimizer_steps"]!=1024
                or result["actual_positive_gradient_steps"]!=1024
                or result["original_source_images"]!=512
                or result["original_h119_derived_sha256"]!=EXPECTED_TRAIN_SHA
                or result["no_EVAL_or_TEST_training"] is not True
                or result["new_strict_GUI_improvement_claimed"] is not False):
            raise ValueError("H119 original actual optimizer receipt not qualified")
        state.update(status=result["status"],actual_verified_optimizer_steps=1024,
                     stage="OWNER_PRIVATE_FINISHED_SHA_PENDING_INDEPENDENT_READBACK",
                     adapter_sha256=result["adapter_sha256"],
                     checkpoint_sha256=result["checkpoint_sha256"],
                     optimizer_sha256=result["optimizer_sha256"])
        record(state)
        print("H119_REAL_1024_ORIGINAL_TRAIN_ONLY_GPU_STEPS_NOT_GUI_SKILL",flush=True)
    except Exception as error:
        state.update(status="FAILED_H119_NO_QUALIFIED_MODEL",
                     failure_class=type(error).__name__)
        record(state)
        print("H119_NOT_QUALIFIED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

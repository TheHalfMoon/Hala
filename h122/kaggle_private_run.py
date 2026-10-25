"""H122 owner-private original TRAIN-only END-grid 448-step free Kaggle T4.

Noncommercial research only. No training on original EVAL, hidden TEST,
or 64 owner-private TRAIN heldout, no release or paid GPU.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="9a5133d254d339d0e3d89993bd7aff886efeb025"
REPO=Path("/kaggle/tmp/hala-h122-frozen-source")
SCRATCH=Path("/kaggle/tmp/hala-h122-original-train448-grid-end")
OUTPUT=Path("/kaggle/working/hala-h122-private-end-grid448-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h122-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)


def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,
                   check=True,timeout=timeout)


def main():
    state={
      "schema":"hala.h122.owner-private-noncommercial-end-absolute-grid448-T4.v1",
      "status":"STARTED","stage":"PRIVATE_FREE_T4_PROVIDER_CHECK",
      "frozen_code_sha":SOURCE_SHA,
      "verified_optimizer_steps":0,
      "paid_resources_used":False,
      "commercial_use":False,
      "hidden_TEST_used":False,
      "no_EVAL_in_training":True,
      "model_weights_published":False,
      "GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private Kaggle only")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("no free Kaggle T4 granted")
        state["stage"]="PINNED_GPU_QLORA_FRAMEWORK"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("source pinned Qwen vision runtime drift")
        state["stage"]="SHA_BOUND_H122_SOURCE"
        record(state)
        if REPO.exists():
            raise ValueError("preexisting source checkout refused")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1",
                "origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],
                                       text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("exact frozen H122 trainer SHA missing")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ORIGINAL_TRAIN448_END_GRID_REAL_OPTIMIZER"
        record(state)
        from h122.research_train import run,STEPS,TRAIN_SHA_H122,HELDOUT_SHA_H122
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result["status"]!="H122_REAL448_TRAIN_ONLY_END_GRID_OPTIMIZER_VERIFIED"
                or result["actual_optimizer_steps"]!=STEPS
                or result["positive_gradient_steps"]!=STEPS
                or result["train_private_sha256"]!=TRAIN_SHA_H122
                or result["heldout_private_sha256"]!=HELDOUT_SHA_H122
                or result["no_EVAL_or_TEST_used"] is not True
                or result["actual_validation_scored"] is not False):
            raise ValueError("H122 actual optimizer and original TRAIN evidence unqualified")
        state.update(status=result["status"],
                     stage="OWNER_PRIVATE_GPU_BINARY_AWAITING_WINDOWS_SHA_REPLAY",
                     verified_optimizer_steps=STEPS,
                     adapter_sha256=result["final_adapter_sha256"],
                     checkpoint_sha256=result["final_checkpoint_sha256"],
                     optimizer_sha256=result["final_optimizer_sha256"])
        record(state)
        print("H122_REAL448_END_GRID_TRAIN_ONLY_NOT_GUI_SKILL",flush=True)
    except Exception as exc:
        state.update(status="FAILED_H122_NO_MODEL_QUALIFICATION",
                     failure_class=type(exc).__name__)
        record(state)
        print("H122_UNQUALIFIED",type(exc).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

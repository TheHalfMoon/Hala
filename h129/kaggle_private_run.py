"""H129 owner-private free Kaggle T4 model-guessed cursor marker pilot."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="e022a5e6f7f92089808b6e0924bd599c7f7547e2"
REPO=Path("/kaggle/tmp/hala-h129-feedback-code")
SCRATCH=Path("/kaggle/tmp/hala-h129-new-original-train16")
OUTPUT=Path("/kaggle/working/hala-h129-private-cursor16")
ATTEMPT=Path("/kaggle/working/hala-h129-cursor16-attempt.json")

def record(obj):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n","utf8")
    temp.replace(ATTEMPT)

def checked(*args,timeout=180):
    proc=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError("H129 pinned source/GPU prerequisite rejected")
    return proc.stdout.strip()

def main():
    state={"schema":"hala.h129.owner-private-pretrained-cursor16.v1",
           "status":"NOT_QUALIFIED","source_git_sha":SOURCE_SHA,
           "stage":"PRIVATE_FREE_T4",
           "model_optimizer_steps":0,"expected_model_calls":64,
           "real_model_calls":0,"paid_compute_used":False,
           "noncommercial":True,"previous_heldout_EVAL_TEST_reused":False,
           "model_skill_qualified":False,
           "is_real_interactive_environment":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H129 requires fresh owner-private Kaggle runtime")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H129 ONLY free Nvidia T4 research resource")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        pinned={"transformers":"5.16.1","torch":"2.11.0","Pillow":"11.3.0"}
        if {key:importlib.metadata.version(key).split("+")[0] for key in pinned}!=pinned:
            raise ValueError("H129 pinned pretrained model runtime drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H129 frozen experiment Git SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="NEW_TRAIN16_REAL_MODEL_GUESSED_CURSOR64"
        record(state)
        from h129.research_gpu import run
        result=run(REPO,SCRATCH,OUTPUT)
        if (result.get("status")!="H129_ACTUAL_PRIVATE_PRETRAINED_CURSOR_MARKER16_SCORED" or
            result.get("real_neural_calls")!=64 or
            result.get("original_train_images_sha_verified")!=16 or
            result.get("old_original_source_ID_and_sha_excluded")!=2704 or
            result.get("new_optimizer_steps")!=0 or
            result.get("original_EVAL_TEST_used") is not False or
            result.get("is_real_interactive_environment") is not False):
            raise ValueError("H129 pilot source-bound result not complete")
        state.update(result,real_model_calls=64)
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H129_ORIGINAL_GUESS_MARKER_PILOT_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

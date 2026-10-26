"""H128 pinned owner-private, noncommercial Kaggle free-T4 GPU pilot."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="36f38361fca8579200eee2990489ca29e12f52db"
REPO=Path("/kaggle/tmp/hala-h128-fresh-zoom16-code")
SCRATCH=Path("/kaggle/tmp/hala-h128-fresh-zoom16-source")
OUTPUT=Path("/kaggle/working/hala-h128-private-zoom16")
ATTEMPT=Path("/kaggle/working/hala-h128-zoom16-attempt.json")

def record(status):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temporary=ATTEMPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n","utf8")
    temporary.replace(ATTEMPT)

def checked(*cmd,timeout=180):
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError("H128 frozen source or free runtime admission rejected")
    return proc.stdout.strip()

def main():
    state={
        "schema":"hala.h128.private-ShowUI2B-zero-shot-anchored-zoom16-attempt.v1",
        "status":"NOT_QUALIFIED","stage":"FREE_T4_GPU",
        "source_git_sha":SOURCE_SHA,
        "original_new_cases":16,"expected_neural_calls":64,
        "real_neural_calls":0,"model_optimizer_steps":0,
        "paid_compute_used":False,"noncommercial":True,
        "prior_EVAL_TEST_or_heldout_used":False,
        "public_original_screenshots_or_weights":False,
        "model_skill_qualified":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H128 fresh owner-private noncommercial Kaggle kernel required")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H128 only free Kaggle Nvidia T4")
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        needed={"transformers":"5.16.1","torch":"2.11.0","Pillow":"11.3.0"}
        if {key:importlib.metadata.version(key).split("+")[0] for key in needed}!=needed:
            raise ValueError("H128 immutable pretrained inference runtime version changed")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin","https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H128 code provenance SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="FRESH_TRAIN16_SOURCE_SHA_AND_64_REAL_MODEL_CALLS"
        record(state)
        from h128.research_gpu import run
        result=run(REPO,SCRATCH,OUTPUT)
        if (result.get("status")!="H128_REAL_PRIVATE_EXPLORATORY_PAIRED_ZOOM16_SCORED"
            or result.get("real_neural_calls")!=64
            or result.get("original_new_images_sha_verified")!=16
            or result.get("original_prior_excluded")!=2688
            or result.get("optimizer_steps")!=0
            or result.get("original_EVAL_TEST_or_prior_heldout_used") is not False):
            raise ValueError("H128 exact private pretrained paired pilot not qualified")
        state.update(result)
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H128_FREE_T4_PAIRED_ZOOM16_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

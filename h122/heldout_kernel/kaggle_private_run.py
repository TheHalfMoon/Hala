"""H122 private 64 original TRAIN-heldout END evaluation (not launched).

Run ONLY after H122 actual 448 optimizer steps and three independent
owner SHA-256 binary pins have been verified and committed to the
frozen evaluator source. Never release hidden source or model weights.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

# This frozen source is intentionally UNARMED: verified model SHA pins
# are None; do not launch against a hypothetical Kaggle training result.
SOURCE_SHA="fe5cab341807897f1a58391d2c47888ba49aaf4b"
REPO=Path("/kaggle/tmp/hala-h122-heldout-frozen-source")
WORK=Path("/kaggle/tmp/hala-h122-private-TRAIN64-END-work")
OUTPUT=Path("/kaggle/working/hala-h122-private-train64-END-eval")
ATTEMPT=Path("/kaggle/working/hala-h122-heldout-attempt.json")


def record(status):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(status,indent=2,sort_keys=True),"utf8")
    tmp.replace(ATTEMPT)


def execute(*args,timeout=180):
    result=subprocess.run(list(args),capture_output=True,text=True,timeout=timeout)
    if result.returncode!=0:
        raise RuntimeError("Pinned prerequisite command refused: "+str(args[:3]))
    return result


def main():
    state={
      "schema":"hala.h122.owner-private-TRAIN64-END-heldout-free-T4.v1",
      "status":"NOT_QUALIFIED",
      "stage":"PRIVATE_KAGGLE_RUNTIME_ADMISSION",
      "frozen_source_sha":SOURCE_SHA,
      "actual_original_heldout_model_calls":0,
      "commercial_use":False,
      "paid_compute_used":False,
      "hidden_TEST_used":False,
      "official_PUBLIC_EVAL_read":False,
      "model_weights_published":False,
      "GUI_skill_qualified":False,
    }
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("H122 owner-private Kaggle noncommercial GPU only")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H122 owner-approved free T4 GPU absent")
        state["stage"]="PINNED_REAL_VISION_INFERENCE_RUNTIME"
        record(state)
        execute(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        execute(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=540)
        required={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {p:importlib.metadata.version(p).split("+")[0] for p in required}!=required:
            raise ValueError("original GPU inference framework versions changed")
        state["stage"]="FROZEN_SOURCE_SHA_CHECK"
        record(state)
        if REPO.exists():
            raise ValueError("preexisting source directory rejected")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        execute("git","init","-q",str(REPO),timeout=30)
        execute("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        execute("git","-C",str(REPO),"fetch","--depth","1",
                "origin",SOURCE_SHA,timeout=180)
        execute("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        exact=execute("git","-C",str(REPO),"rev-parse","HEAD").stdout.strip()
        if exact!=SOURCE_SHA:
            raise ValueError("H122 frozen evaluation code SHA differs")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="OWNER_SHA_VERIFIED_MODEL_AND_64_ORIGINAL_TRAIN_HELDOUT"
        record(state)
        from h122.eval_heldout import run,HOLDOUT
        report=run(repo=REPO,work=WORK,output=OUTPUT)
        if (report.get("status")!="H122_REAL_PRIVATE_ORIGINAL_TRAIN64_HELDOUT_SCORED"
                or report.get("actual_model_inference_calls")!=HOLDOUT
                or report.get("no_fresh_PUBLIC_EVAL_or_hidden_TEST_read") is not True
                or report.get("no_holdout_used_for_training") is not True
                or report.get("no_training_during_eval") is not True):
            raise ValueError("H122 genuine all-64 independent heldout proof incomplete")
        state.update(status=report["status"],
                     stage="INDEPENDENT_WINDOWS_REPLAY_STILL_REQUIRED",
                     actual_original_heldout_model_calls=HOLDOUT,
                     frozen_split_sha=report["frozen_split_sha256"],
                     heldout_gate_passed=report["heldout_gate_passed"],
                     strict_END_hits=report["strict_END_hits"],
                     raw_output_sha256=report["private_raw_model_sha256"])
        record(state)
        print("H122_TRAIN64_REAL_NEURAL_SCORED_NOT_EXTERNAL_GUI_SKILL",flush=True)
    except Exception as exc:
        state.update(status="H122_HELDOUT_NOT_QUALIFIED",failure_class=type(exc).__name__)
        record(state)
        print("H122_HELDOUT_REFUSED",type(exc).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

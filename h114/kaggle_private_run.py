"""H114 isolated 256-step original H113 dual-point visual+language LoRA on free T4.

Never imports or overwrites prior H106 weights; new model checkpoint private.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="8da0ac570561657837e968b9c10a08ae7f216162"
REPO=Path("/kaggle/tmp/hala-h114-source")
SCRATCH=Path("/kaggle/tmp/hala-h114-original-dual-point-train")
OUTPUT=Path("/kaggle/working/hala-h114-private-dual-point256-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h114-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    p=ATTEMPT.with_suffix(".tmp")
    p.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    p.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,check=True,timeout=timeout)

def main():
    state={
       "schema":"hala.h114.private-original-H113-point256-T4-run.v1",
       "status":"STARTED","stage":"OWNER_ONLY_FREE_T4_PREFLIGHT",
       "immutable_source_sha":SOURCE_SHA,
       "actual_dual_point_optimizer_steps_verified":0,
       "model_gui_skill_qualified":False,
       "model_weights_published":False,
       "hidden_eval_or_test_train":False,
       "paid_resources_used":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner private original H113 Kaggle research only")
        import torch
        if not torch.cuda.is_available() or not any(
              "T4" in torch.cuda.get_device_name(i)
              for i in range(torch.cuda.device_count())):
            raise ValueError("private genuine free Kaggle T4 required")
        state["stage"]="EXACT_NONCOMMERCIAL_MODEL_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        current={k:importlib.metadata.version(k).split("+")[0] for k in expected}
        if current!=expected:raise ValueError("pinned original Qwen processor dependency mismatch")
        state["stage"]="SHA_PINNED_H114_NEW_OPT_IN_SOURCE"
        record(state)
        if REPO.exists():raise ValueError("old unqualified checkout refused")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:raise ValueError("H114 source SHA identity failed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="REAL_FRESH_256_DUAL_POINT_LOSS_OPTIMIZER"
        record(state)
        from h114.research_train import run
        result=run(repo=REPO,scratch=SCRATCH,output=OUTPUT)
        if (result["status"]!="H114_REAL_256_ORIGINAL_DUAL_POINT_OPTIMIZER_UPDATES_VERIFIED"
                or result["actual_optimizer_steps"]!=256
                or result["positive_gradient_and_changed_weights_steps"]!=256
                or result["pilot_source_images"]!=128
                or result["start_point_training_rows"]!=128
                or result["end_point_training_rows"]!=128):
            raise ValueError("fresh original 256 point optimizer steps not verified")
        state.update(status=result["status"],
                     stage="OWNER_PRIVATE_DUAL_POINT_CHECKPOINT_RECEIPT",
                     actual_dual_point_optimizer_steps_verified=256,
                     checkpoint_sha256=result["checkpoint_sha256"],
                     adapter_sha256=result["adapter_sha256"],
                     private_point_sft_sha256=result["original_h113_private_dual_point_sha256"])
        record(state)
        print("H114_REAL_ORIGINAL_256_DUAL_POINT_GPU_TRAIN_UPDATES_VERIFIED",flush=True)
    except Exception as exc:
        state.update(status="FAILED_H114_NO_QUALIFIED_POINT_MODEL",
                     failure_class=type(exc).__name__)
        record(state)
        print("H114_PRIVATE_POINT_TRAIN_NOT_QUALIFIED",type(exc).__name__,flush=True)
        raise

if __name__=="__main__":main()

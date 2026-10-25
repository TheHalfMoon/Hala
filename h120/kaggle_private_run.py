"""H120 owner-private proposed new 64-scene strict GUI comparison.

DO NOT PUSH TO KAGGLE UNTIL H119 HAS INDEPENDENT OWNER-VERIFIED
MODEL+CHECKPOINT+OPTIMIZER SHA PINS COMMITTED. Fails closed otherwise.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="5c3b7e53fd8f87b319ba38f554e473e093c41371"
REPO=Path("/kaggle/tmp/hala-h120-frozen-source")
WORK=Path("/kaggle/tmp/hala-h120-next64-paired-official-GUI")
OUTPUT=Path("/kaggle/working/hala-h120-private-original-next64-three-arm")
ATTEMPT=Path("/kaggle/working/hala-h120-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temporary=ATTEMPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    temporary.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,
                   check=True,timeout=timeout)

def main():
    state={"schema":"hala.h120.free-T4-next64-paired-three-model.v1",
           "status":"STARTED","stage":"PRIVATE_FREE_NONCOMMERCIAL_T4",
           "frozen_evaluator_sha":SOURCE_SHA,
           "paid_resources_used":False,"commercial_use":False,
           "model_weights_published":False,
           "general_GUI_skill_qualified":False,
           "hidden_TEST_used":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private Kaggle evaluation only")
        import torch
        if not torch.cuda.is_available() or not any(
             "T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
            raise ValueError("free owner Kaggle T4 not granted")
        state["stage"]="PINNED_QWEN3VL_GPU_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0",
                "Pillow==11.3.0",timeout=540)
        packages={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {key:importlib.metadata.version(key).split("+")[0]
            for key in packages}!=packages:
            raise ValueError("pinned original inference runtime changed")
        state["stage"]="EXACT_FROZEN_H120_SOURCE_SHA"
        record(state)
        if REPO.exists():
            raise ValueError("old unverified evaluator source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1",
                "origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(["git","-C",str(REPO),
                                         "rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("H120 code SHA not exact")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ADMIT_THREE_OWNER_PRIVATE_VERIFIED_CHECKPOINTS"
        record(state)
        from h120.eval import run,ARMS
        result=run(repo=REPO,work=WORK,output=OUTPUT)
        if (result["status"]!="H120_REAL_NEXT64_H114_H117_H119_STRICT_GUI_COMPLETE"
                or result["source_scenes"]!=64
                or set(result["per_arm"])!=set(ARMS)
                or result["unchanged_original_H61_strict_scorer"] is not True
                or result["hidden_TEST_used"] is not False
                or result["training_during_eval"] is not False):
            raise ValueError("H120 original new-64 neural evaluation incomplete")
        state.update(status=result["status"],
                     stage="OWNER_PRIVATE_RAW_NEURAL_OUTPUT_AWAITING_INDEPENDENT_REPLAY",
                     strict_full_drag_successes={
                        arm:result["per_arm"][arm]["task_success"] for arm in ARMS},
                     model_inference_calls=result["model_inference_calls"],
                     raw_model_attempts_sha256=result["private_raw_attempts_sha256"])
        record(state)
        print("H120_64_NEW_ORIGINAL_STRICT_GUI_COMPLETED_NOT_YET_OWNER_REPLAYED",flush=True)
    except Exception as error:
        state.update(status="FAILED_H120_GUI_NOT_QUALIFIED",
                     failure_class=type(error).__name__)
        record(state)
        print("H120_NO_GUI_QUALIFICATION",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

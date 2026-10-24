"""H115 real strict 64-case independent source evaluation, private free T4."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="3c1f86f54552983e2af1dfc9e37f193eb2b866e1"
REPO=Path("/kaggle/tmp/hala-h115-frozen-source")
WORK=Path("/kaggle/tmp/hala-h115-original-eval60-75")
OUTPUT=Path("/kaggle/working/hala-h115-private-64-dual-point-strict")
ATTEMPT=Path("/kaggle/working/hala-h115-attempt.json")

def record(row):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    next_=ATTEMPT.with_suffix(".tmp")
    next_.write_text(json.dumps(row,sort_keys=True,indent=2),"utf8")
    next_.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)

def main():
    state={"schema":"hala.h115.private-64-source-heldout-true-GUI.v1",
           "status":"STARTED","stage":"OWNER_PRIVATE_T4_PREFLIGHT",
           "evaluator_source_sha":SOURCE_SHA,
           "raw_source_or_weights_published":False,
           "model_weights_changed":False,
           "paid_resources_used":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("only owner noncommercial private Kaggle GPU")
        import torch
        if not torch.cuda.is_available() or not any(
              "T4" in torch.cuda.get_device_name(i)
              for i in range(torch.cuda.device_count())):
            raise ValueError("genuine free GPU T4 required")
        state["stage"]="EXACT_QWEN3VL_H106_H114_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        observed={k:importlib.metadata.version(k).split("+")[0] for k in expected}
        if observed!=expected:raise ValueError("unqualified original pinned Qwen stack")
        state["stage"]="EXACT_FROZEN_GITHUB_EVALUATOR_SHA"
        record(state)
        if REPO.exists():raise ValueError("unsafe preexisting model source")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        current=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if current!=SOURCE_SHA:raise ValueError("frozen evaluator commit changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ACTUAL_NEW_64_GUI_SCREENS_TWO_FROZEN_MODEL_ARMS"
        record(state)
        from h115.eval import ARMS,run
        result=run(repo=REPO,work=WORK,output=OUTPUT)
        if (result["status"]!="H115_REAL_NEW_SOURCE_64_H106_VS_H114_STRICT_GUI_COMPLETE"
                or result["sample_counts"]!={arm:64 for arm in ARMS}
                or result["original_h61_strict_scorer_unchanged"] is not True
                or result["eval_labels_not_in_model_prompts"] is not True):
            raise ValueError("new original strict 64 screen GUI evaluation incomplete")
        state.update(status=result["status"],
                     stage="PRIVATE_SOURCE_PINNED_RAW_NEURAL_RESULT",
                     strict_full_drag_successes=result["full_drag_strict_successes"],
                     native_final_actions_valid=result["native_valid_final_action"],
                     raw_neural_sha256=result["raw_model_attempt_sha256"])
        record(state)
        print("H115_NEW_ORIGINAL_64_STRICT_REAL_H106_VS_H114_GUI_COMPLETED",flush=True)
    except Exception as error:
        state.update(status="FAILED_H115_NO_QUALIFIED_FRESH_GUI_RESULT",
                     failure_class=type(error).__name__)
        record(state)
        print("H115_UNQUALIFIED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

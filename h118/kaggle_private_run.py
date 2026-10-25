"""H118 owner-private real 64-scene three-arm GUI comparison, free T4 only."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

# Exact committed, owner-admitted original H117 checkpoint hashes are in this source.
SOURCE_SHA="a3afb2913c476de8ac7af28e5cc98275873316fa"
REPO=Path("/kaggle/tmp/hala-h118-frozen-source")
WORK=Path("/kaggle/tmp/hala-h118-original-eval-76-91")
OUTPUT=Path("/kaggle/working/hala-h118-private-original64-threearm")
ATTEMPT=Path("/kaggle/working/hala-h118-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temporary=ATTEMPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    temporary.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)

def main():
    state={"schema":"hala.h118.private-fresh64-signed-delta-strict.v1",
           "status":"STARTED","stage":"PRIVATE_NONCOMMERCIAL_FREE_T4",
           "frozen_evaluator_sha":SOURCE_SHA,
           "paid_resources_used":False,
           "commercial_use":False,
           "model_weights_changed":False,
           "weights_or_source_media_published":False,
           "model_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner-private Kaggle research only")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("genuine free Kaggle T4 required")
        state["stage"]="PINNED_QWEN3VL_RUNTIME"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("pinned Qwen vision stack changed")
        state["stage"]="EXACT_GITHUB_H118_ADMITTED_SOURCE"
        record(state)
        if REPO.exists():
            raise ValueError("unverified prior source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("H118 frozen evaluator source mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ORIGINAL_NEW_64_SCENES_UNCHANGED_H61_STRICT_THREE_ARMS"
        record(state)
        from h118.eval import ARMS,run
        result=run(repo=REPO,work=WORK,output=OUTPUT)
        if (result["status"]!="H118_REAL_SOURCE_DISJOINT_64_THREE_ARM_STRICT_GUI_COMPLETE"
                or result["sample_counts"]!={a:64 for a in ARMS}
                or result["unchanged_original_H61_strict_scorer"] is not True
                or result["eval_target_boxes_not_used_by_models"] is not True
                or result["hidden_TEST_used"] is not False
                or result["training_performed_in_evaluation"] is not False):
            raise ValueError("original new 64-source strict GUI evaluation incomplete")
        state.update(status=result["status"],
                     stage="OWNER_PRIVATE_NEURAL_OUTPUT_FOR_INDEPENDENT_REPLAY",
                     strict_full_drag_successes=result["strict_full_drag_successes"],
                     native_valid_final_actions=result["native_valid_final_actions"],
                     model_inference_calls=result["model_inference_calls"],
                     private_neural_attempts_sha256=result["private_neural_attempts_sha256"])
        record(state)
        print("H118_REAL_NEW64_THREE_ARM_STRICT_GUI_EVALUATED_NOT_INDEPENDENTLY_REPLAYED",flush=True)
    except Exception as exc:
        state.update(status="FAILED_H118_NO_QUALIFIED_NEW64_GUI_RESULT",
                     failure_class=type(exc).__name__)
        record(state)
        print("H118_UNQUALIFIED",type(exc).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

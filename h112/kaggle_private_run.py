"""H112 private free T4 no-training normalized-vs-source-pixel actual GUI A/B."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="1e3f2269b1bd64d76e4f9289c7d39c00b7417aac"
REPO=Path("/kaggle/tmp/hala-h112-source")
WORK=Path("/kaggle/tmp/hala-h112-eval56-59")
OUTPUT=Path("/kaggle/working/hala-h112-private-relative-vs-pixels")
ATTEMPT=Path("/kaggle/working/hala-h112-attempt.json")


def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    staging=ATTEMPT.with_suffix(".tmp")
    staging.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    staging.replace(ATTEMPT)


def checked(*cmd,timeout=120):
    subprocess.run(list(cmd),check=True,text=True,capture_output=True,timeout=timeout)


def main():
    state={"schema":"hala.h112.private-qwen-native-relative-vs-pixels-attempt.v1",
           "status":"STARTED","stage":"OWNER_PRIVATE_FREE_T4_PREFLIGHT",
           "pinned_source_sha":SOURCE_SHA,"training_executed":False,
           "model_weights_changed":False,"model_weights_published":False,
           "paid_cloud_used":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner private Kaggle noncommercial source bound study only")
        import torch
        if not torch.cuda.is_available() or not any(
          "T4" in torch.cuda.get_device_name(i)
          for i in range(torch.cuda.device_count())):
            raise ValueError("private free T4 accelerator required")
        state["stage"]="EXACT_QWEN3VL_PINNED_T4_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        observed={k:importlib.metadata.version(k).split("+")[0] for k in expected}
        if observed!=expected:raise ValueError("exact private processor/model dependency mismatch")
        state["stage"]="SOURCE_PINNED_GITHUB_FETCH"
        record(state)
        if REPO.exists():raise ValueError("untrusted existing repository")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        head=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if head!=SOURCE_SHA:raise ValueError("H112 exact source code SHA mismatch")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="STRICT_FOUR_ARM_NEW_SOURCE_RELATIVE_COORDINATE_COMPARISON"
        record(state)
        from h112.eval import ARMS,run
        summary=run(repo=REPO,work=WORK,output=OUTPUT)
        if (summary["status"]!="H112_REAL_SOURCE_DISJOINT_QWEN_RELATIVE_AB_COMPLETE"
                or summary["sample_counts"]!={arm:16 for arm in ARMS}
                or summary["original_H61_strict_scorer_unchanged"] is not True
                or summary["native_relative_outputs_mapped_before_original_strict_scorer"] is not True):
            raise ValueError("relative output or source benchmark qualification failed")
        state.update(status=summary["status"],
                     stage="OWNER_PRIVATE_RAW_NEURAL_ATTEMPT_EVIDENCE",
                     strict_gui_success=summary["native_strict_action_successes"],
                     native_json_valid=summary["native_strict_action_json_valid"],
                     raw_attempt_sha256=summary["raw_model_attempt_sha256"])
        record(state)
        print("H112_ORIGINAL_NATIVE_RELATIVE_VS_PIXEL_REAL_GUI_EVAL_COMPLETE",flush=True)
    except Exception as error:
        state.update(status="FAILED_H112_NO_QUALIFIED_GUI_GROUNDING_RESULT",
                     failure_class=type(error).__name__)
        record(state)
        print("H112_UNQUALIFIED",type(error).__name__,flush=True)
        raise


if __name__=="__main__":main()

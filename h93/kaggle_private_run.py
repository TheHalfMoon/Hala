"""Private Kaggle H93 paired official-source held-out diagnostic. No training."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="362b81374a6fcf74d45cd666434d74f94895f8a0"
REPO=Path("/kaggle/tmp/hala-h93-source")
WORK=Path("/kaggle/tmp/hala-h93-public-eval-scratch")
OUT=Path("/kaggle/working/hala-h93-private-eval")
ATTEMPT=Path("/kaggle/working/hala-h93-attempt.json")


def process(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)


def save(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding="utf8")
    tmp.replace(ATTEMPT)


def run():
    state={"schema":"hala.h93.private-official-eval-attempt.v1",
           "status":"STARTED","stage":"KAGGLE_RESEARCH_CUDA",
           "source_sha":SOURCE_SHA,"model_trained_during_evaluation":False,
           "evaluation_complete":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("genuine private Kaggle research session required")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<1:
            raise RuntimeError("real free CUDA GPU required")
        state["gpu_name"]=torch.cuda.get_device_name(0)
        state["stage"]="PINNED_QWEN_DEPENDENCIES"
        process(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        process(sys.executable,"-m","pip","install","-q","transformers==5.16.1",
                "peft==0.20.0","Pillow==11.3.0",timeout=480)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if any(importlib.metadata.version(k).split("+")[0]!=v
               for k,v in expected.items()):
            raise ValueError("unpinned model inference stack")
        state["stage"]="PINNED_H93_EVALUATION_CODE"
        if REPO.exists():
            raise FileExistsError("refusing existing unverified source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        process("git","init","-q",str(REPO))
        process("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        process("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=150)
        process("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=30)
        actual=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("evaluation source SHA mismatched")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="OFFICIAL_HELDOUT_BASE_AND_H92_INFERENCE"
        save(state)
        from h93.eval import run as evaluate
        result=evaluate(repo=REPO,work=WORK,output=OUT)
        if (result.get("status")!="REAL_PAIRED_OFFICIAL_PUBLIC_EVAL_DIAGNOSTIC"
                or result.get("sample_counts",{}).get("base",0)==0
                or result["sample_counts"]["base"]!=result["sample_counts"]["h92"]):
            raise ValueError("actual paired evaluation not qualified")
        state.update(stage="COMPLETED_OFFICIAL_PAIRED_DIAGNOSTIC",
                     status="PAIRED_BASE_H92_INFERENCE_COMPLETE",
                     evaluation_complete=True,
                     cases=result["sample_counts"]["base"],
                     base_task_successes=result["base_successes"],
                     h92_task_successes=result["h92_successes"],
                     score_receipt="hala-h93-private-eval/h93-private-eval-summary.json")
        save(state)
        print("HALA_H93_OFFICIAL_EVAL_COMPLETE",json.dumps({
            "cases":state["cases"],"base":state["base_task_successes"],
            "h92":state["h92_task_successes"]}),flush=True)
    except Exception as e:
        state.update(status="FAILED_UNQUALIFIED_EVALUATION",
                     failure_class=type(e).__name__)
        save(state)
        print("HALA_H93_DIAGNOSTIC_BLOCKED",type(e).__name__,flush=True)
        raise

if __name__=="__main__":
    run()

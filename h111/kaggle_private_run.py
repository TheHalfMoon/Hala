"""H111 free private Kaggle T4 pilot: original vs model-proposed zoom crops."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path

SOURCE_SHA="bf5c09f1680b1840b9223a5917ec52122c3661e1"
REPO=Path("/kaggle/tmp/hala-h111-source")
WORK=Path("/kaggle/tmp/hala-h111-original-eval48-55")
OUTPUT=Path("/kaggle/working/hala-h111-private-focus512-eval")
ATTEMPT=Path("/kaggle/working/hala-h111-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    staging=ATTEMPT.with_suffix(".tmp")
    staging.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    staging.replace(ATTEMPT)

def checked(*cmd,timeout=120):
    subprocess.run(list(cmd),check=True,capture_output=True,text=True,timeout=timeout)

def main():
    state={"schema":"hala.h111.original-source-frozen-model-crop-ab.v1",
           "status":"STARTED","stage":"OWNER_ONLY_T4_PREFLIGHT",
           "exact_source_sha":SOURCE_SHA,
           "new_training_executed":False,"model_weights_changed":False,
           "paid_resources":False,"commercial_use":False,
           "model_published":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("owner private free Kaggle research only")
        import torch
        if not torch.cuda.is_available() or not any(
             "T4" in torch.cuda.get_device_name(i)
             for i in range(torch.cuda.device_count())):
            raise ValueError("free T4 not available")
        state["stage"]="EXACT_PINNED_GPU_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        required={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in required}!=required:
            raise ValueError("Qwen3-VL frozen processor dependencies mismatch")
        state["stage"]="EXACT_H111_RESEARCH_GIT_SHA"
        record(state)
        if REPO.exists():raise ValueError("preexisting unverified Git source")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:raise ValueError("H111 source commit not source-bound")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="EXACT_H106_FROZEN_PAIRED_ORIGINAL_VS_ROI_512_INFERENCE"
        record(state)
        from h111.eval import run,ARMS
        result=run(repo=REPO,work=WORK,output=OUTPUT)
        if (result["status"]!="H111_REAL_PAIRED_CONFIRMATION_32_MODEL_CROPS_COMPLETE"
                or result["source_cases"]!=32
                or set(result["strict_full_drag_successes"])!=set(ARMS)
                or result["original_source_boxes_not_used_during_inference"] is not True):
            raise ValueError("paired actual source-disjoint focus study incomplete")
        state.update(status=result["status"],
                     stage="PRIVATE_SOURCE_BOUND_RAW_NEURAL_EVIDENCE",
                     source_cases=32,
                     model_refinement_calls=result["other_diagnostics"]["refinement_calls"],
                     strict_successes=result["strict_full_drag_successes"],
                     raw_sha256=result["raw_neural_sha256"])
        record(state)
        print("H111_FRESH_ORIGINAL_SOURCE_MODEL_FOCUS_EVAL_RECORDED",flush=True)
    except Exception as error:
        state.update(status="FAILED_H111_RESEARCH_EVAL_GATE",
                     failure_class=type(error).__name__)
        record(state)
        print("H111_PRIVATE_RESEARCH_NOT_QUALIFIED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":main()

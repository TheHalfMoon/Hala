"""Owner-private H124 original TRAIN128 neural full drag test, no training."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path

SOURCE_SHA="ca0c2e646255a4c6b4926972b5c94ce6b413763e"
REPO=Path("/kaggle/tmp/hala-h124-eval-frozen-source")
WORK=Path("/kaggle/tmp/hala-h124-private-new-TRAIN128")
OUTPUT=Path("/kaggle/working/hala-h124-private-joint-train128-full-drag")
ATTEMPT=Path("/kaggle/working/hala-h124-heldout-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)

def execute(*args,timeout=180):
    process=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if process.returncode:
        raise RuntimeError("H124 pinned private GPU runtime or frozen source rejected")
    return process.stdout.strip()

def main():
    state={
      "schema":"hala.h124.owner-private-real-TRAIN128-joint-full-drag.v1",
      "source_sha":SOURCE_SHA,"status":"NOT_QUALIFIED",
      "stage":"OWNER_FREE_T4_MODEL_ADMISSION",
      "real_model_calls":0,"training_during_evaluation":False,
      "original_EVAL_or_TEST_used":False,"heldout_trained_on":False,
      "paid_compute_used":False,"commercial_use":False,
      "weights_published":False,"external_GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("H124 only private noncommercial Kaggle")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H124 free Kaggle T4 unavailable")
        state["stage"]="PINNED_GPU_PROCESSOR_MODEL";record(state)
        execute(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        execute(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {key:importlib.metadata.version(key).split("+")[0]
            for key in expected}!=expected:
            raise ValueError("H124 model inference framework pins differ")
        if REPO.exists():
            raise ValueError("H124 original source repo already exists")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        state["stage"]="EXACT_OWNER_INDEPENDENT_MODEL_BYTE_SHA";record(state)
        execute("git","init","-q",str(REPO))
        execute("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        execute("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        execute("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if execute("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H124 exact private source commit differs")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="NEW_ORIGINAL_128_TRAIN_HELDOUT_NEURAL_INFERENCE";record(state)
        from h124.eval_heldout import run,HOLDOUT
        result=run(REPO,WORK,OUTPUT)
        if (result.get("status")!=
              "H124_REAL_PRIVATE_ORIGINAL_TRAIN128_FULL_DRAG_SCORED"
              or result.get("actual_model_inference_calls")!=HOLDOUT
              or result.get("original_EVAL_or_TEST_used") is not False
              or result.get("heldout_used_for_optimizer") is not False
              or result.get("training_during_eval") is not False):
            raise ValueError("H124 genuine 128 model outputs/target source not scored")
        state.update(
          status=result["status"],real_model_calls=HOLDOUT,
          heldout_gate_passed=result["heldout_gate_passed"],
          strict_START_hits=result["strict_original_START_hits"],
          strict_END_hits=result["strict_original_END_hits"],
          strict_full_drag_hits=result["strict_original_full_drag_hits"],
          raw_neural_sha=result["private_raw_neural_sha256"],
          stage="OWNER_WINDOWS_NEURAL_SOURCE_REPLAY_REQUIRED")
        record(state)
    except Exception as error:
        state.update(status="FAILED_H124_NEURAL_HELDOUT_UNQUALIFIED",
                     failure_class=type(error).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

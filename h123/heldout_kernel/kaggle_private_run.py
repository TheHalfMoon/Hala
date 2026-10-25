"""H123 owner-private original TRAIN128 END evaluation; trained weights SHA verified."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path

SOURCE_SHA="3d7c305ed23dd9509529def8f76f599efc021ede"
REPO=Path("/kaggle/tmp/hala-h123-heldout-source")
WORK=Path("/kaggle/tmp/hala-h123-heldout-original-TRAIN128")
OUTPUT=Path("/kaggle/working/hala-h123-private-original-train128-END-eval")
ATTEMPT=Path("/kaggle/working/hala-h123-heldout-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True),"utf8")
    tmp.replace(ATTEMPT)

def execute(*args,timeout=180):
    proc=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if proc.returncode:
        raise RuntimeError("H123 private pinned runtime refused")
    return proc.stdout.strip()

def main():
    state={
        "schema":"hala.h123.owner-private-original-TRAIN128-heldout-T4.v1",
        "source_sha":SOURCE_SHA,"status":"NOT_QUALIFIED",
        "stage":"FREE_PRIVATE_GPU_ADMISSION",
        "real_model_inference_calls":0,
        "no_public_EVAL_or_TEST":True,"heldout_not_trained":True,
        "paid_compute_used":False,"commercial_use":False,
        "weights_published":False,"GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("H123 free owner Kaggle required")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
            raise ValueError("H123 free T4 GPU absent")
        state["stage"]="PINNED_GPU_FRAMEWORK";record(state)
        execute(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        execute(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("H123 framework version mismatches frozen inference")
        if REPO.exists():
            raise ValueError("H123 preexisting private source checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        state["stage"]="PINNED_ACTUAL_OWNER_MODEL_BI_SHA";record(state)
        execute("git","init","-q",str(REPO))
        execute("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        execute("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        execute("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if execute("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H123 evaluation code source SHA changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        from h123.eval_heldout import run,HOLDOUT
        state["stage"]="REAL_128_UNSEEN_ORIGINAL_TRAIN_SOURCE_INFERENCE";record(state)
        report=run(REPO,WORK,OUTPUT)
        if (report.get("status")!=
                "H123_REAL_PRIVATE_ORIGINAL_TRAIN128_HELDOUT_SCORED"
                or report.get("actual_model_inference_calls")!=HOLDOUT
                or report.get("original_EVAL_or_TEST_used") is not False
                or report.get("heldout_used_for_optimizer") is not False
                or report.get("training_during_eval") is not False):
            raise ValueError("H123 full private original TRAIN128 neural evidence incomplete")
        state.update(status=report["status"],real_model_inference_calls=HOLDOUT,
                     heldout_gate_passed=report["heldout_gate_passed"],
                     strict_END_hits=report["strict_END_hits"],
                     raw_output_sha256=report["private_raw_neural_sha256"],
                     stage="OWNER_INDEPENDENT_WINDOWS_SOURCE_NEURAL_REPLAY_PENDING")
        record(state)
    except Exception as error:
        state.update(status="H123_ORIGINAL_TRAIN128_HELDOUT_NOT_QUALIFIED",
                     failure_class=type(error).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

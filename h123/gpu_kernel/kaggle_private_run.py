"""H123 owner-private FREE noncommercial T4 real TRAIN896; no publication."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path
SOURCE_SHA="90bdaeb83e83bb269426043e28f2dddd335c79fb"
REPO=Path("/kaggle/tmp/hala-h123-private-gpu-source")
WORK=Path("/kaggle/tmp/hala-h123-private-complex896")
OUTPUT=Path("/kaggle/working/hala-h123-private-complex896-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h123-gpu-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    temp.replace(ATTEMPT)

def execute(*args,timeout=180):
    process=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if process.returncode:
        raise RuntimeError("H123 pinned runtime source/prerequisite refusal")
    return process.stdout.strip()

def main():
    state={"schema":"hala.h123.private-real896-complex-dualgrid-T4.v1",
      "status":"NOT_QUALIFIED","stage":"FREE_NONCOMMERCIAL_T4",
      "source_git_sha":SOURCE_SHA,"actual_optimizer_steps":0,
      "paid_compute_used":False,"commercial_use":False,
      "original_EVAL_or_TEST_used":False,"heldout_trained_on":False,
      "weights_published":False,"GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("Kaggle private session required")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
            raise ValueError("no free Kaggle T4 allocated")
        state["stage"]="PINNED_QWEN_GPU_DEPENDENCIES";record(state)
        execute(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        execute(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in expected}!=expected:
            raise ValueError("H123 framework pins drift")
        if REPO.exists():
            raise ValueError("refuse preexisting frozen source")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        execute("git","init","-q",str(REPO))
        execute("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        execute("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        execute("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if execute("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("frozen source SHA changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="REAL_ORIGINAL_H71_TRAIN896_FRESH";record(state)
        from h123.research_train import run,STEPS,TRAIN_SHA,HELDOUT_SHA
        data=run(repo=REPO,scratch=WORK,output=OUTPUT)
        if (data["status"]!="H123_REAL896_NEW_COMPLEX_TRAIN_ONLY_OPTIMIZER_VERIFIED"
                or data["actual_train_optimizer_steps"]!=STEPS
                or data["positive_gradient_steps"]!=STEPS
                or data["train_private_sha256"]!=TRAIN_SHA
                or data["heldout_private_sha256"]!=HELDOUT_SHA
                or data["previous_H113_H122_source_reused"] is not False
                or data["heldout_scored_or_used_for_optimizer"] is not False):
            raise ValueError("H123 actual original TRAIN optimizer evidence invalid")
        state.update(status=data["status"],actual_optimizer_steps=STEPS,
            stage="INDEPENDENT_OWNER_BINARY_REPLAY_REQUIRED",
            adapter_sha256=data["final_adapter_sha256"],
            checkpoint_sha256=data["final_checkpoint_sha256"],
            optimizer_sha256=data["final_optimizer_sha256"])
        record(state)
    except Exception as e:
        state.update(status="FAILED_H123_NO_GUI_MODEL_QUALIFICATION",
                     failure_class=type(e).__name__)
        record(state)
        raise
if __name__=="__main__":
    main()

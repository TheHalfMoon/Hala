"""H124 fresh 896-step joint START END signed delta, owner private T4."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path

SOURCE_SHA="c76d93be56c5897f3881c8556db8f55491156a80"
REPO=Path("/kaggle/tmp/hala-h124-fresh-joint-source")
WORK=Path("/kaggle/tmp/hala-h124-fresh-joint896")
OUTPUT=Path("/kaggle/working/hala-h124-private-new-joint896-checkpoint")
ATTEMPT=Path("/kaggle/working/hala-h124-gpu-attempt.json")

def record(value):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)

def run_cmd(*args,timeout=180):
    completed=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if completed.returncode:
        raise RuntimeError("H124 pinned private GPU runtime or source refused")
    return completed.stdout.strip()

def main():
    state={
      "schema":"hala.h124.owner-private-real896-joint-start-end-T4.v1",
      "status":"NOT_QUALIFIED","stage":"OWNER_FREE_NONCOMMERCIAL_GPU",
      "source_git_sha":SOURCE_SHA,"actual_optimizer_steps":0,
      "paid_compute_used":False,"commercial_use":False,
      "original_EVAL_or_TEST_used":False,"heldout_optimizer_input":False,
      "weights_published":False,"GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H124 only owner Kaggle noncommercial T4")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H124 no free Kaggle T4")
        state["stage"]="EXACT_QWEN_T4_FRAMEWORK";record(state)
        run_cmd(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        run_cmd(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","peft==0.20.0","Pillow==11.3.0",
                timeout=540)
        expected={"torch":"2.11.0","transformers":"5.16.1",
                  "peft":"0.20.0","Pillow":"11.3.0"}
        if {name:importlib.metadata.version(name).split("+")[0]
             for name in expected}!=expected:
            raise ValueError("H124 frozen vision runtime version drift")
        if REPO.exists():
            raise ValueError("H124 source must be new immutably pinned checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        state["stage"]="FROZEN_ORIGINAL_SOURCE_SHA";record(state)
        run_cmd("git","init","-q",str(REPO))
        run_cmd("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        run_cmd("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,
                timeout=180)
        run_cmd("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if run_cmd("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H124 code source SHA changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="TRUE_FRESH_ORIGINAL_TRAIN896_SOURCE_JOINT";record(state)
        from h124.research_train import run,STEPS,TRAIN_SHA,HELDOUT_SHA
        receipt=run(REPO,WORK,OUTPUT)
        if (receipt.get("status")!=
              "H124_REAL896_FRESH_JOINT_TRAIN_ONLY_OPTIMIZER_VERIFIED"
              or receipt.get("actual_optimizer_steps")!=STEPS
              or receipt.get("positive_gradient_steps")!=STEPS
              or receipt.get("train_private_sha256")!=TRAIN_SHA
              or receipt.get("heldout_private_sha256")!=HELDOUT_SHA
              or receipt.get("old_H105_H123_images_excluded")!=1536
              or receipt.get("heldout_used_for_optimizer") is not False
              or receipt.get("original_EVAL_or_TEST_in_training") is not False):
            raise ValueError("H124 genuine joint model optimizer receipt incomplete")
        state.update(status=receipt["status"],
                     stage="OWNER_WINDOWS_INDEPENDENT_896_BINARY_REPLAY_REQUIRED",
                     actual_optimizer_steps=STEPS,
                     adapter_sha256=receipt["final_adapter_sha256"],
                     checkpoint_sha256=receipt["final_checkpoint_sha256"],
                     optimizer_sha256=receipt["final_optimizer_sha256"])
        record(state)
    except Exception as err:
        state.update(status="FAIL_H124_REAL_TRAINING_NOT_QUALIFIED",
                     failure_class=type(err).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

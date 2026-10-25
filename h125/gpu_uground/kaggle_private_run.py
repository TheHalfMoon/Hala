"""H125 owner PRIVATE free noncommercial T4 pretrained GUI zero-shot, one arm.

This is EVALUATION ONLY: 0 optimizer steps and no original source or
model weights published. Exact 64 NEW H71 original TRAIN scenes each arm.
"""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path
SOURCE_SHA="d7671635517508820e56e5f5661185159e6c2fea"
ARM="uground_2b"
REPO=Path("/kaggle/tmp/hala-h125-frozen-gui2b-"+ARM)
WORK=Path("/kaggle/tmp/hala-h125-new-64-source-"+ARM)
OUTPUT=Path("/kaggle/working/hala-h125-private-"+ARM+"-gui64")
ATTEMPT=Path("/kaggle/working/hala-h125-"+ARM+"-attempt.json")

def record(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    tmp.replace(ATTEMPT)

def checked(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H125 exact private source/GPU runtime admission rejected")
    return result.stdout.strip()

def main():
    state={
      "schema":"hala.h125.owner-private-native-GUI2B-zero-shot64-v1",
      "status":"NOT_QUALIFIED","stage":"PRIVATE_FREE_T4_RUNTIME",
      "source_git_sha":SOURCE_SHA,"model_arm":ARM,
      "original_source_cases":64,"expected_model_calls":128,
      "real_model_calls":0,"optimizer_steps":0,
      "original_EVAL_or_TEST_used":False,"old_heldout_reused":False,
      "paid_compute_used":False,"commercial_use":False,
      "model_or_screenshots_published":False,
      "external_GUI_skill_qualified":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H125 only actual owner Kaggle private run")
        import torch
        if not torch.cuda.is_available() or not any(
            "T4" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())):
            raise ValueError("H125 real free T4 GPU unavailable")
        state["stage"]="FROZEN_QWEN2_GUI_PRETRAINED_GPU_RUNTIME";record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        required={"transformers":"5.16.1","torch":"2.11.0","Pillow":"11.3.0"}
        if {k:importlib.metadata.version(k).split("+")[0] for k in required}!=required:
            raise ValueError("H125 exact free T4 ML runtime SHA/version changed")
        if REPO.exists():
            raise ValueError("H125 source cannot reuse previous mutable checkout")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        state["stage"]="SOURCE_SHA_QUALIFIED_NEW_TRAIN64";record(state)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H125 exact frozen source Git SHA changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HALA_REMOTE_TRAINING_ALLOWED="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="REAL_NATIVE_GUI_ZERO_SHOT_64X2_ENDPOINTS";record(state)
        from h125.model_contracts import MODELS
        from h125.research_eval import run,TOTAL
        if ARM not in MODELS:
            raise ValueError("H125 unknown pinned model arm")
        receipt=run(REPO,WORK,OUTPUT,ARM)
        if (receipt.get("status")!="H125_REAL_PRIVATE_NEW64_PRETRAINED_GUI2B_SCORED"
            or receipt.get("real_neural_endpoint_calls")!=TOTAL*2
            or receipt.get("original_new_image_byte_SHA_verified")!=TOTAL
            or receipt.get("model_id")!=MODELS[ARM].model_id
            or receipt.get("model_revision")!=MODELS[ARM].revision_sha
            or receipt.get("no_finetuning") is not True
            or receipt.get("model_train_steps")!=0
            or receipt.get("prior_H123_H124_heldout_reused") is not False
            or receipt.get("original_EVAL_or_TEST_used") is not False):
            raise ValueError("H125 original GUI model 128 native outputs incomplete")
        state.update(
          status=receipt["status"],stage="OWNER_RAW_64_GROUNDING_SHA_REPLAY_REQUIRED",
          real_model_calls=TOTAL*2,
          native_valid_both=receipt["native_valid_both_points"],
          strict_START_hits=receipt["strict_original_START_hits"],
          strict_END_hits=receipt["strict_original_END_hits"],
          strict_full_drag_hits=receipt["strict_original_full_drag_hits"],
          internal_gate_passed=receipt["internal_gate_passed"],
          raw_neural_sha256=receipt["private_raw_sha256"],
          model_revision=receipt["model_revision"])
        record(state)
    except Exception as error:
        state.update(status="FAIL_H125_UNQUALIFIED_GUI_NEURAL_ARM",
                     failure_class=type(error).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

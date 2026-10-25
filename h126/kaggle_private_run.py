"""H126 owner private noncommercial CPU source/task-definition audit only."""
from pathlib import Path
import json,os,subprocess,sys

SOURCE_SHA="f3df7c7544c09dbbacb0e32d160092a4082ee04e"
REPO=Path("/kaggle/tmp/hala-h126-source-only-frozen")
SCRATCH=Path("/kaggle/tmp/hala-h126-original-train-64")
OUTPUT=Path("/kaggle/working/hala-h126-original-TRAIN64-definition-audit.json")
ATTEMPT=Path("/kaggle/working/hala-h126-private-source-attempt.json")

def record(obj):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True),"utf8")
    temp.replace(ATTEMPT)

def checked(*args,timeout=180):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("H126 original private source runtime or checkout rejected")
    return result.stdout.strip()

def main():
    state={
      "schema":"hala.h126.owner-private-original-TRAIN64-definition-CPU.v1",
      "status":"NOT_QUALIFIED","stage":"CPU_ONLY_FROZEN_TRAIN_SOURCE",
      "source_git_sha":SOURCE_SHA,
      "excluded_previous_source_count":2624,
      "model_optimizer_steps":0,"model_inference_calls":0,
      "paid_compute_used":False,"commercial_use":False,
      "original_EVAL_or_TEST_used":False,
      "original_source_or_weights_public":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H126 only owner private free Kaggle CPU")
        if REPO.exists():
            raise ValueError("H126 owner original frozen checkout must be fresh")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H126 original research code SHA mismatch")
        sys.path.insert(0,str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        state["stage"]="ORIGINAL_TRAIN64_MEDIA_AND_INTENT_SHA_AUDIT"
        record(state)
        from h126.research_cpu import run
        data=run(SCRATCH,OUTPUT)
        if (data.get("status")!=
            "PASS_H126_NEW_ORIGINAL_TRAIN64_SOURCE_LANGUAGE_GEOMETRY_CPU"
            or data.get("actual_original_image_byte_sha_verified")!=64
            or data.get("excluded_all_prior_2624_source_images")!=2624
            or data.get("semantic_identifiability_inferred") is not False
            or data.get("no_previous_2624_screenshots_used") is not True
            or data.get("new_neural_inference_calls")!=0
            or data.get("new_optimizer_steps")!=0):
            raise ValueError("H126 genuine original source definition audit failed")
        state.update(status=data["status"],new_original_source_cases=64,
                     stage="SOURCE_QUALIFIED_SEMANTIC_SUFFICIENCY_NOT_YET_ESTABLISHED",
                     new_source_schedule_sha=data["schedule_sha256"])
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H126_SOURCE_DEFINITION_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

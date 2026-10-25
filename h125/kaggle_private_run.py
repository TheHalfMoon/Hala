"""H125 owner private FREE Kaggle CPU genuine new64 source admission only."""
import json,os,subprocess,sys
from pathlib import Path
SOURCE_SHA="b41843e4d82a4cff2ecc2268b735340971e38526"
REPO=Path("/kaggle/tmp/hala-h125-original-new64-frozen-source")
WORK=Path("/kaggle/tmp/hala-h125-original-gui64-source")
OUTPUT=Path("/kaggle/working/hala-h125-new-original64-gui-source-receipt.json")
ATTEMPT=Path("/kaggle/working/hala-h125-cpu-source-attempt.json")

def record(value):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True),"utf8")
    temp.replace(ATTEMPT)

def execute(*args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("H125 exact source or runtime refused")
    return p.stdout.strip()

def main():
    state={
      "schema":"hala.h125.owner-private-original-new64-CPU-only.v1",
      "source_git_sha":SOURCE_SHA,"status":"NOT_QUALIFIED",
      "stage":"OWNER_CPU_ONLY_ORIGINAL_NEW64",
      "model_inference_calls":0,"optimizer_steps":0,
      "original_EVAL_TEST_used":False,
      "paid_compute_used":False,"commercial_use":False,
      "weights_or_original_media_published":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H125 only authorized owner private Kaggle CPU")
        if REPO.exists():
            raise ValueError("H125 original source SHA checkout cannot be reused")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        execute("git","init","-q",str(REPO))
        execute("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        execute("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        execute("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if execute("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H125 actual SHA-frozen H125 source differs")
        sys.path.insert(0,str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        state["stage"]="ALL_PREVIOUS_2560_SOURCE_EXCLUDED_AND_TRUE_DUAL_BOX_SHA"
        record(state)
        from h125.research_cpu import run,TOTAL
        receipt=run(WORK,OUTPUT)
        if (receipt.get("status")!="PASS_OWNER_H125_NEW_64_ORIGINAL_TRAIN_GUI_SOURCE"
                or receipt.get("genuine_image_bytes_and_both_endpoints_verified")!=TOTAL
                or receipt.get("prior_original_screenshots_excluded")!=2560
                or receipt.get("prior_training_or_heldout_images_reused") is not False
                or receipt.get("optimizer_steps")!=0
                or receipt.get("model_inference_calls")!=0):
            raise ValueError("H125 64 original genuine new TRAIN screenshots unavailable")
        state.update(status=receipt["status"],
                     stage="ORIGINAL_SOURCE_QUALIFIED_NO_GPU_OR_MODEL",
                     original_source_cases=TOTAL,
                     schedule_sha256=receipt["schedule_sha256"])
        record(state)
    except Exception as exc:
        state.update(status="FAILED_H125_SOURCE_UNQUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise

if __name__=="__main__":
    main()

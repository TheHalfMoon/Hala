"""H124 owner-private CPU real H71 original TRAIN joint START+END source check."""
import json,os,subprocess,sys
from pathlib import Path

SOURCE_SHA="2ae21c21f9ff061cb2ee9f16c244f045f4627d53"
REPO=Path("/kaggle/tmp/hala-h124-frozen-source")
SCRATCH=Path("/kaggle/tmp/hala-h124-original-train-cpu")
OUTPUT=Path("/kaggle/working/hala-h124-original-new1024-source-receipt.json")
ATTEMPT=Path("/kaggle/working/hala-h124-source-attempt.json")

def write(state):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    file=ATTEMPT.with_suffix(".tmp")
    file.write_text(json.dumps(state,sort_keys=True,indent=2),"utf8")
    file.replace(ATTEMPT)

def command(*args,timeout=180):
    p=subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
    return p.stdout.strip()

def main():
    state={"schema":"hala.h124.owner-private-CPU-new-original-joint1024.v1",
      "status":"UNQUALIFIED","stage":"OWNER_FREE_CPU_SOURCE_SHA",
      "source_git_sha":SOURCE_SHA,
      "model_train_steps":0,"model_inference_calls":0,
      "paid_compute_used":False,"original_EVAL_TEST_used":False,
      "source_or_weights_published":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise ValueError("H124 owner-only free Kaggle CPU")
        if REPO.exists():
            raise ValueError("H124 source checkout must be create-only")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(REPO))
        command("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(REPO),"fetch","--depth","1",
                "origin",SOURCE_SHA,timeout=180)
        command("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H124 frozen source commit mismatch")
        sys.path.insert(0,str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        state["stage"]="TRUE_H71_NEW_TRAIN1024_DUAL_BOX_PROOF"
        write(state)
        from h124.research_cpu import run
        receipt=run(SCRATCH,OUTPUT)
        if (receipt.get("status")!=
                "PASS_OWNER_H124_NEW_ORIGINAL_TRAIN1024_JOINT_GEOMETRY_SOURCE"
                or receipt.get("original_image_byte_and_two_box_SHA_verified")!=1024
                or receipt.get("new_train_images")!=896
                or receipt.get("new_heldout_images")!=128
                or receipt.get("all_previous_H105_H123_images_excluded")!=1536
                or receipt.get("no_original_EVAL_TEST") is not True):
            raise ValueError("H124 genuine new original START/END TRAIN source unavailable")
        state.update(status="PASS_H124_REAL_ORIGINAL_SOURCE_ONLY",
                     train_images=896,heldout_images=128,
                     schedule_sha256=receipt["schedule_sha256"],
                     stage="OWNER_SOURCE_SHA_CHECKED_NO_GPU")
        write(state)
    except Exception as e:
        state.update(status="FAIL_H124_NO_SOURCE_MODEL_QUALIFICATION",
                     failure_class=type(e).__name__)
        write(state)
        raise
if __name__=="__main__":
    main()

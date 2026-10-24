"""H102 immutable H71 TRAIN geometry audit on FREE owner-private Kaggle CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="7c978a5147d04c36be51c999f543e80d5485df71"
REPO=Path("/kaggle/tmp/hala-h102-source")
SCRATCH=Path("/kaggle/tmp/hala-h102-immutable-h71")
OUTPUT=Path("/kaggle/working/hala-h102-private-train-audit")
ATTEMPT=Path("/kaggle/working/hala-h102-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    candidate=ATTEMPT.with_suffix(".tmp")
    candidate.write_text(json.dumps(data,sort_keys=True,indent=2),"utf8")
    candidate.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),text=True,capture_output=True,
                   check=True,timeout=timeout)

def main():
    state={"schema":"hala.h102.owner-private-source-geometry-cpu-attempt.v1",
           "status":"STARTED","stage":"CPU_PREFLIGHT","source_sha":SOURCE_SHA,
           "gpu_training_executed":False,"model_weights_downloaded":False,
           "private_source_data_published":False,"commercial_use":False}
    try:
        if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or not Path("/kaggle/working").is_dir()):
            raise ValueError("owner private Kaggle required")
        # The runtime contains only this runner before GitHub source checkout.
        # The pinned H73 CPU-only admission runs again inside h102.train_audit.
        import shutil
        Path("/kaggle/tmp").mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage("/kaggle/tmp").free<5*1024**3:
            raise OSError(">=5GB Kaggle CPU scratch needed to rederive original H71")
        state["stage"]="PINNED_PRIVATE_SOURCE_FETCH"
        record(state)
        if REPO.exists():
            raise ValueError("preexisting unqualified source")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        actual=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],
                                        text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("CPU H102 source commit differs from pinned")
        sys.path.insert(0,str(REPO))
        state["stage"]="OWNER_PRIVATE_H71_10K_REDERIVATION_AND_128_MEDIA_AUDIT"
        record(state)
        from h102.train_audit import run
        result=run(SCRATCH,OUTPUT)
        state.update(status=result["status"],
                     stage="OWNER_PRIVATE_TRAIN_GEOMETRY_RECEIPT",
                     source_rows=result["original_source_rows_scanned"],
                     h99_media_samples=result["actual_H99_original_samples_checked"],
                     invalid_label_counts=result["original_10k_label_failures"])
        record(state)
        print("H102_ORIGINAL_H71_CPU_TRAIN_GEOMETRY_GATE",result["status"],flush=True)
        if result["status"]!="TRAIN_LABEL_GEOMETRY_GATE_PASS":
            raise ValueError("source geometry audit found invalid TRAIN labels")
    except Exception as error:
        state["status"]="FAILED_H102_SOURCE_OR_GEOMETRY_GATE"
        state["failure_type"]=type(error).__name__
        record(state)
        print("H102_TRAIN_AUDIT_BLOCKED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":
    main()

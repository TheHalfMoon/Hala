"""Private free Kaggle CPU-only H103 Qwen3-VL processor/mask probe."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

SOURCE_SHA="9b3c064e5dcfce954c45913dc654ab9a946769a9"
REPO=Path("/kaggle/tmp/hala-h103-source")
SCRATCH=Path("/kaggle/tmp/hala-h103-original-h71")
OUT=Path("/kaggle/working/hala-h103-private-processor-mask")
ATTEMPT=Path("/kaggle/working/hala-h103-attempt.json")

def record(data):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    candidate=ATTEMPT.with_suffix(".tmp")
    candidate.write_text(json.dumps(data,sort_keys=True,indent=2),"utf8")
    candidate.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),capture_output=True,text=True,check=True,timeout=timeout)

def main():
    state={"schema":"hala.h103.owner-private-cpu-only-processor-probe.v1",
           "source_sha":SOURCE_SHA,"status":"STARTED","stage":"PRIVATE_CPU_PREFLIGHT",
           "GPU_used":False,"model_weights_downloaded":False,
           "training_executed":False,"model_published":False,
           "private_data_published":False,"commercial_use":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("approved personal noncommercial private Kaggle required")
        Path("/kaggle/tmp").mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage("/kaggle/tmp").free<5*1024**3:
            raise OSError("not enough ephemeral free disk for original H71")
        state["stage"]="PINNED_CPU_PROCESSOR_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        versions={"transformers":"5.16.1","Pillow":"11.3.0"}
        if any(importlib.metadata.version(k)!=v for k,v in versions.items()):
            raise ValueError("pinned processor dependency drift")
        state["stage"]="EXACT_GIT_COMMIT_SOURCE"
        record(state)
        if REPO.exists():
            raise ValueError("unverified existing checkout forbidden")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=45)
        actual=subprocess.check_output(
            ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if actual!=SOURCE_SHA:
            raise ValueError("H103 processor/source code changed from pinned")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="ORIGINAL_H71_8_CASE_PIXEL_AND_MASK_PROBE"
        record(state)
        from h103.processor_audit import run
        result=run(SCRATCH,OUT)
        if (result["status"]!="PINNED_H71_QWEN3VL_PROCESSOR_MASK_GATE_PASS"
                or result["audited_original_bilingual_family_screenshots"]!=8
                or result["training_executed"] is not False):
            raise ValueError("actual processor-masking qualification failed")
        state.update(status="H103_ACTUAL_PINNED_PROCESSOR_MASK_8_CASE_PASS",
                     stage="PRIVATE_AGGREGATE_RECEIPT_READY",
                     source_sha256=result["source_h71_sha256"],
                     audited_cases=8)
        record(state)
        print("H103_ACTUAL_ORIGINAL_PROCESSOR_MASK_PASS",flush=True)
    except Exception as error:
        state["status"]="FAILED_H103_PROCESSOR_MASK_QUALIFICATION"
        state["failure_class"]=type(error).__name__
        record(state)
        print("H103_PRIVATE_CPU_PROBE_BLOCKED",type(error).__name__,flush=True)
        raise

if __name__=="__main__":main()

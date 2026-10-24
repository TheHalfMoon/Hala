"""H104 private Kaggle CPU actual Qwen3-VL processor coordinate bridge oracle."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_SHA="37c0d26e6308a6966e4a24f6fdfb4eaf52c5dff3"
REPO=Path("/kaggle/tmp/hala-h104-source")
SCRATCH=Path("/kaggle/tmp/hala-h104-private-original-H71")
OUTPUT=Path("/kaggle/working/hala-h104-private-processor-bridge")
ATTEMPT=Path("/kaggle/working/hala-h104-attempt.json")

def record(s):
    ATTEMPT.parent.mkdir(parents=True,exist_ok=True)
    temp=ATTEMPT.with_suffix(".tmp")
    temp.write_text(json.dumps(s,sort_keys=True,indent=2),"utf8")
    temp.replace(ATTEMPT)

def checked(*args,timeout=120):
    subprocess.run(list(args),check=True,capture_output=True,text=True,timeout=timeout)

def main():
    state={"schema":"hala.h104.private-real-processor-geometry-bridge-cpu.v1",
           "source_sha":SOURCE_SHA,"stage":"CPU_PREFLIGHT","status":"STARTED",
           "gpu_used":False,"training_executed":False,
           "model_weights_downloaded":False,"commercial_use":False,
           "private_source_material_published":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not Path("/kaggle/working").is_dir():
            raise ValueError("private owner noncommercial Kaggle runtime only")
        Path("/kaggle/tmp").mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage("/kaggle/tmp").free<5*1024**3:
            raise OSError("original H71 extraction requires private ephemeral CPU scratch")
        state["stage"]="PINNED_CPU_PROCESSOR_DEPENDENCIES"
        record(state)
        checked(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        checked(sys.executable,"-m","pip","install","-q",
                "transformers==5.16.1","Pillow==11.3.0",timeout=540)
        if (importlib.metadata.version("transformers")!="5.16.1"
                or importlib.metadata.version("Pillow")!="11.3.0"):
            raise ValueError("processor dependency pin drift")
        state["stage"]="PINNED_GITHUB_SOURCE"
        record(state)
        if REPO.exists():raise ValueError("preexisting source checkout denied")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO),timeout=30)
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git",timeout=30)
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA,timeout=180)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD",timeout=40)
        got=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
        if got!=SOURCE_SHA:raise ValueError("H104 exact source SHA differs")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["stage"]="REAL_ORIGINAL_H71_PROCESSOR_BRIDGE_AND_H61_ORACLE"
        record(state)
        from h104.actual_train_bridge import run
        result=run(SCRATCH,OUTPUT)
        state.update(status=result["status"],
                     original_H71_cases=result["original_H71_samples"],
                     source_to_processor_frame_changes=result["resized_cases"],
                     unchanged_H61_strict_oracle_successes=result["strict_original_H61_roundtrip_success"])
        record(state)
        if result["status"]!="ORIGINAL_8_TRAIN_PROCESSOR_BRIDGE_STRICT_ORACLE_PASS":
            raise ValueError("original source strict roundtrip mapping gate failed")
        print("H104_ACTUAL_ORIGINAL_TRAIN_BRIDGE_STRICT_ORACLE_PASS",flush=True)
    except Exception as error:
        state["status"]="FAILED_H104_ORIGINAL_PROCESSOR_BRIDGE_ADMISSION"
        state["failure_class"]=type(error).__name__
        record(state)
        print("H104_ORIGINAL_PRIVATE_SOURCE_BRIDGE_BLOCKED",type(error).__name__,flush=True)
        raise
if __name__=="__main__":main()

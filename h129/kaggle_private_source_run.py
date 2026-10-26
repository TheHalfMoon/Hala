"""H129 independent owner-private Kaggle free CPU original-TRAIN16 source export."""
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_SHA="__PIN_SOURCE_SHA__"
REPO=Path("/kaggle/tmp/hala-h129-source16-cpu-replay")
SCRATCH=Path("/kaggle/tmp/hala-h129-source16-cpu-staging")
OUTPUT=Path("/kaggle/working/hala-h129-private-original16-source.json")
ATTEMPT=Path("/kaggle/working/hala-h129-source16-attempt.json")

def record(obj):
    tmp=ATTEMPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n","utf8")
    tmp.replace(ATTEMPT)
def checked(*args):
    proc=subprocess.run(args,capture_output=True,text=True,timeout=180)
    if proc.returncode:
        raise RuntimeError("H129 fresh owner CPU source Git SHA acquisition failed")
    return proc.stdout.strip()

def main():
    state={"schema":"hala.h129.private-source16-CPU-owner-replay-attempt.v1",
           "status":"NOT_QUALIFIED","source_git_sha":SOURCE_SHA,
           "optimizer_steps":0,"model_inference_calls":0,
           "paid_compute_used":False,"noncommercial":True,
           "original_EVAL_TEST_or_previous_heldout_used":False}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H129 private fresh Kaggle CPU kernel required")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        checked("git","init","-q",str(REPO))
        checked("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        checked("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_SHA)
        checked("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if checked("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_SHA:
            raise ValueError("H129 CPU original source code SHA mismatch")
        sys.path.insert(0,str(REPO))
        from h73.kaggle_h73_cpu import require_cpu
        require_cpu()
        from h129.source_cpu import run
        state.update(stage="FROZEN_SOURCE16_PRIVATE_SHA_EXPORT")
        record(state)
        result=run(SCRATCH,OUTPUT)
        if (result.get("status")!="PASS_H129_OWNER_PRIVATE_TRAIN16_SOURCE_REPLAY_CPU" or
            result.get("source_verified")!=16 or
            result.get("excluded_previous")!=2688):
            raise ValueError("H129 original independent owner source export failed")
        state.update(result)
        record(state)
    except Exception as exc:
        state.update(status="FAIL_H129_PRIVATE_CPU_SOURCE_REPLAY_NOT_QUALIFIED",
                     failure_class=type(exc).__name__)
        record(state)
        raise
if __name__=="__main__":
    main()

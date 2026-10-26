"""H137 pinned one-shot PRIVATE FREE T4 144-call cross-library benchmark.

Frozen H136 adapter; exact 36 independent MIT vendor-backed UI screenshots.
Model weights never trained or published in this evaluation kernel.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_GIT_SHA="10197bae6f63cecd1f57be4bcffd9db93fa21277"
REPO=Path("/kaggle/tmp/hala-h137-source-once")
OUTPUT=Path("/kaggle/working/hala-h137-private-crosslib36")
RECEIPT=Path("/kaggle/working/hala-h137-private-crosslib-attempt.json")

def save(data):
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    f=RECEIPT.with_suffix(".tmp")
    f.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    f.replace(RECEIPT)

def command(*args,timeout=180):
    completed=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if completed.returncode:
        raise RuntimeError("H137 frozen, private T4 prerequisite failed")
    return completed.stdout.strip()

def main():
    state={"schema":"hala.h137.one-shot-owner-private-eval-attempt.v1",
        "status":"NOT_QUALIFIED",
        "source_git_sha":SOURCE_GIT_SHA,
        "actual_optimizer_steps":0,
        "paid_compute_used":False,
        "release_qualified":False,
        "main_Hala_flagship_updated":False,
        "owner_private_benchmark":True}
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or REPO.exists():
            raise ValueError("H137 private noncommercial fresh Kaggle only")
        import torch
        if not torch.cuda.is_available() or not any(
                "T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())):
            raise ValueError("H137 FREE NVIDIA T4 required; no paid fallback")
        root=Path("/kaggle/input")
        if not root.is_dir() or any(
            "train" in p.name.lower() or "h136-private-dev" in p.name.lower()
            for p in root.iterdir() if p.is_dir()):
            raise ValueError("H137 evaluation-only sources; reject training/dev mounts")
        command(sys.executable,"-m","pip","uninstall","-y","torchao",timeout=90)
        command(sys.executable,"-m","pip","install","-q",
             "transformers==5.17.0","peft==0.21.2",
             "Pillow==11.3.0",timeout=540)
        required={"transformers":"5.17.0","peft":"0.21.2",
                  "torch":"2.11.0","Pillow":"11.3.0"}
        observed={k:importlib.metadata.version(k).split("+")[0] for k in required}
        if observed!=required:
            raise ValueError("H137 pinned model runtime version drift")
        REPO.parent.mkdir(parents=True,exist_ok=True)
        command("git","init","-q",str(REPO))
        command("git","-C",str(REPO),"remote","add","origin",
                "https://github.com/TheHalfMoon/Hala.git")
        command("git","-C",str(REPO),"fetch","--depth","1","origin",SOURCE_GIT_SHA)
        command("git","-C",str(REPO),"checkout","--detach","-q","FETCH_HEAD")
        if command("git","-C",str(REPO),"rev-parse","HEAD")!=SOURCE_GIT_SHA:
            raise ValueError("H137 frozen Git revision changed")
        sys.path.insert(0,str(REPO))
        os.environ.update(HALA_NONCOMMERCIAL_RESEARCH="1",
                          HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
        state["status"]="RUNNING_144_FROZEN_NEURAL_EVALS"
        save(state)
        from h137.gpu_eval import run
        evidence=run(OUTPUT)
        if (evidence.get("status")!="H137_PRIVATE_THREE_REAL_MIT_INTERACTION_LIBRARIES_144_INFERENCE" or
            evidence.get("actual_neural_calls")!=144 or
            evidence.get("actual_optimizer_steps")!=0 or
            evidence.get("paid_compute_used") is not False):
            raise ValueError("H137 genuine foreign library full inference run missing")
        state.update(evidence)
        save(state)
    except Exception as exc:
        state.update(status="FAIL_H137_NO_QUALIFIED_NEURAL_EVAL",
                     failure_class=type(exc).__name__)
        save(state)
        raise
if __name__=="__main__":main()

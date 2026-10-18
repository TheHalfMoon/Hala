"""Create a private, GPU-only H25 research notebook; never submit a job.

Saved Kaggle Versions may ONLY be submitted after review/CI and explicit
confirmation that the allocation is personally permitted and non-billable.
"""
import argparse
import json
from pathlib import Path

from h25 import baseline

KERNEL = "abdulazizshehri/hala-h25-mind-dev-first-20261018"
BRANCH = "ps/h25-baseline-admission"


def cell(kind, text):
    output = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        output.update(execution_count=None, outputs=[])
    return output


def generate(source_commit, target):
    if not isinstance(source_commit, str) or not baseline.SOURCE_SHA.fullmatch(source_commit):
        raise ValueError("One exact immutable source SHA40 required")
    target = Path(target)
    if target.exists():
        raise FileExistsError("Prospective notebook bundle cannot overwrite prior evidence")
    target.mkdir(parents=True, exist_ok=False)
    name = "hala-h25-mind-dev.ipynb"
    kernel = {
        "id": KERNEL, "title": "Hala H25 first bilingual Mind DEV baseline",
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    admission = """import os
from pathlib import Path
if os.name != "posix" or not Path("/kaggle/working").is_dir():
    raise RuntimeError("Non-Kaggle notebook is not authorized")
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "Batch":
    raise RuntimeError("Only retained Kaggle Batch notebooks may run")
if (Path("/kaggle/working/hala-h25-research-src").exists()
        or Path("/kaggle/working/hala-h25-dev-out").exists()):
    raise FileExistsError("Research directory or result already exists")
print("HALA_H25_PRIVATE_FREE_GPU_BATCH_REQUIRED")
"""
    deps = """%pip install --quiet --no-deps transformers==5.16.1 accelerate==1.12.0 safetensors==0.8.0 Pillow==11.3.0
import importlib.metadata
required={"torch":"2.11.0+cu128","transformers":"5.16.1",
          "accelerate":"1.12.0","safetensors":"0.8.0","Pillow":"11.3.0"}
for pkg, ver in required.items():
    if importlib.metadata.version(pkg) != ver:
        raise RuntimeError("Unreviewed remote dependency "+pkg)
"""
    source = f"""import os, subprocess
from pathlib import Path
root=Path("/kaggle/working/hala-h25-research-src")
subprocess.run(["git","clone","--quiet","--depth","1","--single-branch",
               "--branch","{BRANCH}","https://github.com/TheHalfMoon/Hala.git",
               str(root)],check=True,timeout=300)
actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
if actual != "{source_commit}":
    raise RuntimeError("Reviewed immutable H25 source branch has moved")
if subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],
                           cwd=root,text=True).strip():
    raise RuntimeError("Unreviewed modifications in H25 remote source")
os.environ["HALA_REMOTE_DEVELOPMENT_ALLOWED"]="1"
os.environ["PYTHONIOENCODING"]="utf-8"
print("HALA_H25_PINNED_SOURCE",actual)
"""
    measure = f"""import subprocess, sys
from pathlib import Path
root=Path("/kaggle/working/hala-h25-research-src")
command=[sys.executable,"-u","-m","h25.baseline",
         "--source-commit","{source_commit}",
         "--output","/kaggle/working/hala-h25-dev-out"]
subprocess.run(command,cwd=root,check=True,timeout=5400)
print("HALA_H25_DEV_RUN_TERMINATED; EXTERNAL RAW HASH AUDIT REQUIRED")
"""
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "hala_source_commit":source_commit,
            "hala_source_branch":BRANCH,
            "hala_protocol":baseline.PROTOCOL,
            "hala_split":"dev",
            "hala_model_adopted":False,
            "hala_gpu_submitted":False,
        },
        "cells":[
            cell("markdown", "# Hala H25 — unchanged Mind baseline / prospective public DEV only\n\n"
                 "Private Kaggle Save & Run All research. No TEST, no training or Hala/Orcel "
                 "device execution. 42 public synthetic DEV cases, 21 English / 21 Arabic. "
                 "Run only after independent review and confirmation of genuinely free hardware; "
                 "the source code generator does not allocate a GPU.\n"),
            cell("code",admission),
            cell("code",deps),
            cell("code",source),
            cell("code",measure),
        ]
    }
    (target/name).write_text(json.dumps(nb,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (target/"kernel-metadata.json").write_text(
        json.dumps(kernel,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"notebook":str(target/name),"metadata":str(target/"kernel-metadata.json"),
            "sha40":source_commit,"submitted":False,"inference_run":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    print(json.dumps(generate(a.source_commit,a.output),sort_keys=True))


if __name__=="__main__":
    main()

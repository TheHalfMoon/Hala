"""Generate private, GPU-only prospective H24 DEV baseline Save & Run All notebook.

This generator never submits a notebook or allocates compute. The source
commit is supplied only after exact-head review; never use a mutable branch.
"""
import argparse
import json
from pathlib import Path
import re

SHA = re.compile(r"^[a-f0-9]{40}$")
KERNEL = "abdulazizshehri/hala-h24-mind-dev-baseline-20261017"


def cell(kind, source):
    record = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        record.update(execution_count=None, outputs=[])
    return record


def generate(source_commit, target):
    if not SHA.fullmatch(source_commit):
        raise ValueError("Immutable reviewed full source SHA40 is required")
    target = Path(target)
    if target.exists():
        raise FileExistsError("New output directory; do not overwrite a saved version")
    target.mkdir(parents=True, exist_ok=False)
    output = target / "h24-mind-dev.ipynb"
    metadata = {"id": KERNEL, "title": "Hala H24 Mind DEV baseline 20261017",
                "code_file": output.name, "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    source = f"""import os
from pathlib import Path
assert os.name == "posix" and Path("/kaggle/working").is_dir()
assert os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"
assert not Path("/kaggle/working/hala-h24-src").exists()
assert not Path("/kaggle/working/hala-h24-mind-dev-output").exists()
print("HALA_H24_FREE_KAGGLE_BATCH_ADMITTED")
"""
    package = """%pip install --quiet --no-deps transformers==5.16.1 accelerate==1.12.0 safetensors==0.8.0 Pillow==11.3.0
import importlib.metadata
assert importlib.metadata.version("torch") == "2.11.0+cu128"
for name, version in {"transformers": "5.16.1", "accelerate": "1.12.0",
                      "safetensors": "0.8.0", "Pillow": "11.3.0"}.items():
    assert importlib.metadata.version(name) == version
"""
    prepare = f"""import os, subprocess, sys
from pathlib import Path
source = Path("/kaggle/working/hala-h24-src")
subprocess.run(["git","clone","--quiet","https://github.com/TheHalfMoon/Hala.git",str(source)],
               check=True, timeout=300)
subprocess.run(["git","checkout","--detach","{source_commit}"],
               cwd=source, check=True, timeout=60, stdout=subprocess.DEVNULL)
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
assert head == "{source_commit}"
assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],
                                   cwd=source,text=True).strip()
os.environ["HALA_REMOTE_DEVELOPMENT_ALLOWED"]="1"
os.environ["PYTHONIOENCODING"]="utf-8"
print("HALA_H24_SOURCE_SHA",head)
"""
    execute = f"""import subprocess,sys
from pathlib import Path
source=Path("/kaggle/working/hala-h24-src")
args=[sys.executable,"-u","-m","h24.mind_dev",
      "--source-commit","{source_commit}",
      "--output","/kaggle/working/hala-h24-mind-dev-output"]
subprocess.run(args, cwd=source, check=True, timeout=5400)
print("H24_DEV_ONLY_RUN_TERMINATED; RETRIEVE RAW OUTPUT BEFORE CLAIMING QUALITY")
"""
    n = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                     "hala_h24_source_commit": source_commit,
                     "hala_h24_split": "dev", "hala_model_adopted": False},
        "cells": [
            cell("markdown", "# Hala H24: unchanged base HalaMind on NEW bilingual DEV\n\n"
                 "Private personal-research Kaggle free T4 Save & Run All only. "
                 "20 DEV cases, 10 English / 10 Arabic. No TEST, no training, "
                 "no Orcel execution, no billing upgrade, no model adoption. "
                 "Retain exact raw per-case outputs and externally verify downloaded results.\n"),
            cell("code", source),
            cell("code", package),
            cell("code", prepare),
            cell("code", execute),
        ]
    }
    output.write_text(json.dumps(n, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    (target/"kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2)+"\n",encoding="utf-8")
    return {"source_commit":source_commit,"notebook":str(output),
            "metadata":str(target/"kernel-metadata.json")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(generate(args.source_commit,args.output),sort_keys=True))


if __name__=="__main__":
    main()

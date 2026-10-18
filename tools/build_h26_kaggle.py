"""Prepare private Kaggle H26 DEV-only notebook; generation never submits GPU."""
import argparse
import json
import re
from pathlib import Path

from h26 import choice,runner

KERNEL_ID="abdulazizshehri/hala-h26-choice-dev-20261018"
BRANCH="ps/h26-typed-choice"


def cell(kind,source):
    item={"cell_type":kind,"metadata":{},"source":source.splitlines(keepends=True)}
    if kind=="code":
        item.update(execution_count=None,outputs=[])
    return item


def generate(source_commit,output):
    if type(source_commit) is not str or not re.fullmatch("[a-f0-9]{40}",source_commit):
        raise ValueError("H26 immutable reviewed source SHA40 required")
    target=Path(output)
    if target.exists():
        raise FileExistsError("Never overwrite old H26 notebook/evidence")
    target.mkdir(parents=True,exist_ok=False)
    meta={
        "id":KERNEL_ID,"title":"hala-h26-choice-dev-20261018",
        "code_file":"hala-h26.ipynb","language":"python",
        "kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
        "dataset_sources":[],"kernel_sources":[],
        "competition_sources":[],"model_sources":[],
    }
    preflight='''import os
from pathlib import Path
if os.name!="posix" or not Path("/kaggle/working").is_dir():
    raise RuntimeError("H26 is Kaggle-only")
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE")!="Batch":
    raise RuntimeError("H26 requires a retained Kaggle Batch version")
for folder in ("hala-h26-research-src","hala-h26-choice-out"):
    if Path("/kaggle/working",folder).exists():
        raise FileExistsError("Cannot reuse any H26 result/source directory")
print("HALA_H26_PRIVATE_DEV_ONLY_ADMITTED")
'''
    deps='''%pip install --quiet --no-deps transformers==5.16.1 accelerate==1.12.0 safetensors==0.8.0 Pillow==11.3.0
import importlib.metadata
expected={"torch":"2.11.0+cu128","transformers":"5.16.1",
          "accelerate":"1.12.0","safetensors":"0.8.0","Pillow":"11.3.0"}
for name,ver in expected.items():
    if importlib.metadata.version(name)!=ver:
        raise RuntimeError("H26 unreviewed package "+name)
'''
    source=f'''import os,subprocess
from pathlib import Path
root=Path("/kaggle/working/hala-h26-research-src")
subprocess.run(["git","clone","--quiet","--depth","1",
                "--single-branch","--branch","{BRANCH}",
                "https://github.com/TheHalfMoon/Hala.git",str(root)],
               check=True,timeout=300)
sha=subprocess.check_output(["git","rev-parse","HEAD"],
                            cwd=root,text=True).strip()
if sha!="{source_commit}":
    raise RuntimeError("H26 reviewed source branch changed")
if subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],
                           cwd=root,text=True).strip():
    raise RuntimeError("Unreviewed source changes")
os.environ["HALA_REMOTE_DEVELOPMENT_ALLOWED"]="1"
os.environ["PYTHONIOENCODING"]="utf-8"
print("H26_EXACT_RESEARCH_SHA",sha)
'''
    run=f'''import subprocess,sys
from pathlib import Path
root=Path("/kaggle/working/hala-h26-research-src")
args=[sys.executable,"-u","-m","h26.runner",
      "--source-commit","{source_commit}",
      "--output","/kaggle/working/hala-h26-choice-out"]
subprocess.run(args,cwd=root,check=True,timeout=5400)
print("H26_PRIVATE_DEV_COMPLETE_ONLY_IF_RAW_AUDIT_PASSES")
'''
    notebook={
        "nbformat":4,"nbformat_minor":5,
        "metadata":{
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "hala_protocol":runner.RUN_PROTOCOL,
            "hala_data_sha256":runner.EXACT_DATA_SHA,
            "hala_source_commit":source_commit,
            "hala_split":"dev",
            "hala_training":False,
            "hala_model_adopted":False,
            "hala_submitted":False,
        },
        "cells":[
            cell("markdown","# Hala H26 private choice-8 DEV research\n\n"
                 "First-party NEW synthetic DEV families, not a protected TEST. "
                 "42 bilingual tasks, no model training, no native Orcel permissions, "
                 "no host PC weights, no billed GPU tier. "
                 "Submit once only after signed CI/review; measure decision choice not mere structured syntax.\n"),
            cell("code",preflight),cell("code",deps),cell("code",source),cell("code",run),
        ],
    }
    (target/"hala-h26.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (target/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"source_sha40":source_commit,"private":True,
            "submitted":False,"model_inference_performed":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-commit",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(generate(args.source_commit,args.output),sort_keys=True))


if __name__=="__main__":
    main()

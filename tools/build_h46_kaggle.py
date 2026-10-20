"""Build private, create-only, explicitly opted-in Kaggle DEV runner.

No network or GPU during generation. Remote run is independent of original
H23 frozen-CUDA experiment and must never retry that failed notebook.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h34.runner import MODEL,REVISION,PINS
from h46.experiment import PROTOCOL,PROMPT_SHA,SOURCE_BRANCH,cases,sha

KAGGLE_ID="abdulazizshehri/hala-h46-8-dev-qwen3vl-pixel-20261020"


def inventory(root:Path,head:str)->dict:
    root=Path(root)
    if not re.fullmatch("[0-9a-f]{40}",head):
        raise ValueError("exact H46 signed commit required")
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=root,timeout=40)
    if (git("rev-parse","HEAD").decode().strip()!=head or
        git("status","--porcelain","--untracked-files=all").strip()):
        raise ValueError("Source checkout must be clean exact H46 head")
    paths=["h46/experiment.py","h46/__init__.py","tools/build_h46_kaggle.py",
           "h34/runner.py","h34/protocol.py","h44/assets/model_inputs.json"]
    paths += ["h44/assets/"+row["screenshot"] for row in cases()]
    hashes={}
    for name in paths:
        blob=(root/name).read_bytes()
        if git("show",f"{head}:{name}")!=blob:
            raise ValueError("Source differs from H46 signed commit: "+name)
        hashes[name]=hashlib.sha256(blob).hexdigest()
    return hashes


def cell(kind:str,body:str)->dict:
    x={"cell_type":kind,"metadata":{},"source":body.splitlines(keepends=True)}
    if kind=="code":x.update(execution_count=None,outputs=[])
    return x


def generate(root:Path,commit:str,target:Path,*,research_allowed:bool=False)->dict:
    checks=inventory(root,commit)
    target=Path(target)
    if target.exists():raise FileExistsError("H46 Kaggle source payload must be create-only")
    target.mkdir(parents=True)
    bootstrap="""import os,json,platform,subprocess,hashlib,time
from pathlib import Path
begin=time.monotonic()
source=Path('/kaggle/working/hala-h46-source')
output=Path('/kaggle/working/hala-h46-real-dev-result')
os.environ['HALA_H46_NONCOMMERCIAL_FREE_RESEARCH']=__ALLOWED__
if os.environ['HALA_H46_NONCOMMERCIAL_FREE_RESEARCH']!='1':
    raise RuntimeError('Explicit H46 free noncommercial research opt-in required')
if (platform.system()!='Linux' or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch'
    or not Path('/kaggle/working').is_dir()):
    raise RuntimeError('Only private free Kaggle Linux Batch accepted')
if source.exists() or output.exists():raise FileExistsError('Refuse original H46 evidence overwrite')
gpu=subprocess.run(['nvidia-smi','-L'],text=True,capture_output=True,timeout=20)
if gpu.returncode!=0 or 'T4' not in gpu.stdout:
    raise RuntimeError('No actual free T4 allocation - fail closed before download')
os.environ.update(HF_HOME='/tmp/hala-h34-cache',
                  HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  DO_NOT_TRACK='1',WANDB_DISABLED='true',
                  TOKENIZERS_PARALLELISM='false')
subprocess.run(['git','clone','--quiet','--depth','1','--single-branch',
                '--branch',__BRANCH__,
                'https://github.com/TheHalfMoon/Hala.git',str(source)],
                timeout=240,check=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=source,text=True,timeout=30).strip()
if git('rev-parse','HEAD')!=__SHA__ or git('status','--porcelain','--untracked-files=all'):
    raise RuntimeError('H46 exact signed source is not clean')
for name,digest in __FILES__.items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Pinned H46 source changed: '+name)
print('H46_FREE_T4_AND_MODEL_ONLY_SOURCE_PREFLIGHT_PASS',__SHA__,len(__FILES__),flush=True)
"""
    for marker,value in {
        "__ALLOWED__":repr("1" if research_allowed else "0"),
        "__BRANCH__":repr(SOURCE_BRANCH),
        "__SHA__":repr(commit),
        "__FILES__":repr(checks)
    }.items():
        bootstrap=bootstrap.replace(marker,value)
    execute="""import os,signal,subprocess,sys,time,json,hashlib
from pathlib import Path
p=subprocess.Popen([sys.executable,'-u','-m','h46.experiment',
   '--source-commit',__SHA__,'--output',str(output)],
   cwd=source,start_new_session=True)
try:
    code=p.wait(timeout=900)
except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:code=p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL)
        code=p.wait(timeout=15)
    raise RuntimeError('H46 external bound reached; do not hide incomplete evidence')
report={'schema':'hala.h46.batch-parent.v1',
  'source_commit':__SHA__,'worker_exit':code,
  'raw_metadata_present':(output/'run_metadata.json').exists(),
  'raw_journal_present':(output/'raw_attempts.jsonl').exists(),
  'completed_unqualified':False,
  'blind_benchmark':False,'model_qualified':False,
  'native_actions':False,
  'elapsed_seconds':round(time.monotonic()-begin,3)}
Path('/kaggle/working/h46-parent-exit.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\\n')
print('H46_WORKER_EXIT',code,'RESULTS_REQUIRE_EXTERNAL_AUDIT',flush=True)
if code!=0:raise RuntimeError('H46 worker failed; preserve original incomplete record')
""".replace("__SHA__",repr(commit))
    metadata={"nbformat":4,"nbformat_minor":5,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "hala_protocol":PROTOCOL,"hala_model":MODEL,"hala_revision":REVISION,
                    "hala_prompt_sha256":PROMPT_SHA,"source_sha":commit,
                    "declared_expected_public_dev_cases":8,
                    "research_optin":research_allowed,
                    "blind_benchmark":False,"model_qualified":False,
                    "commercial":False,"native_actions":False},
        "cells":[
            cell("markdown","# Hala H46 original eight-case bilingual synthetic DEV grounding\n"
                 "Private noncommercial free Kaggle T4 only. Exact Qwen3VL 2B. "
                 "No H23 CUDA lifecycle, independent heldout leaderboard or OS actions.\n"),
            cell("code",bootstrap),
            cell("code",execute)]}
    name="hala-h46-original-dev-qwen3vl.ipynb"
    (target/name).write_text(json.dumps(metadata,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    kaggle={"id":KAGGLE_ID,"title":"Hala H46 Eight Original Bilingual DEV Screenshots",
            "code_file":name,"language":"python","kernel_type":"notebook",
            "is_private":True,"enable_gpu":True,"enable_tpu":False,
            "enable_internet":True,"dataset_sources":[],"competition_sources":[],
            "kernel_sources":[],"model_sources":[]}
    (target/"kernel-metadata.json").write_text(json.dumps(kaggle,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"source_commit":commit,"source_file_count":len(checks),
            "private":True,"expected_cases":8,"submitted":False,
            "research_allowed":research_allowed,"output":str(target)}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--source-commit",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--research-allowed",action="store_true")
    a=ap.parse_args()
    print(json.dumps(generate(a.root,a.source_commit,a.output,research_allowed=a.research_allowed),sort_keys=True))

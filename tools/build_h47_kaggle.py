"""Create-only private free Kaggle research notebook for exactly the H47 paired prompts.

Requires exact clean signed commit and all original model-input blob hashes.
No submission, GPU, API or paid cost from constructing it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h47.experiment import SOURCE_BRANCH,PROTOCOL,PROMPT_SHA,plan,ROOT
from h34.runner import MODEL,REVISION

KAGGLE_ID="abdulazizshehri/hala-h47-paired-prompt-dev-eight"
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def source_inventory(root:Path,commit:str)->dict:
    if re.fullmatch("[0-9a-f]{40}",commit) is None:raise ValueError("Exact signed H47 source required")
    def git(*args):return subprocess.check_output(["git",*args],cwd=root,timeout=40)
    if (git("rev-parse","HEAD").decode().strip()!=commit or
        git("status","--porcelain","--untracked-files=all").strip()):
        raise ValueError("H47 notebook only from clean immutable source")
    paths=["h47/__init__.py","h47/experiment.py","tools/build_h47_kaggle.py",
           "h46/experiment.py","h36/protocol.py","h34/runner.py",
           "h34/protocol.py","h44/assets/model_inputs.json"]
    from h46.experiment import cases
    paths+=["h44/assets/"+x["screenshot"] for x in cases()]
    out={}
    for name in paths:
        blob=(root/name).read_bytes()
        if git("show",f"{commit}:{name}")!=blob:
            raise ValueError("Original signed H47 source bytes changed: "+name)
        out[name]=sha(blob)
    return out
def cell(kind:str,source:str)->dict:
    x={"cell_type":kind,"metadata":{},"source":source.splitlines(keepends=True)}
    if kind=="code":x.update(outputs=[],execution_count=None)
    return x
def generate(root:Path,commit:str,target:Path,*,research_allowed:bool=False)->dict:
    source=source_inventory(root,commit)
    target=Path(target)
    if target.exists():raise FileExistsError("H47 private original notebook destination exists")
    target.mkdir(parents=True)
    init="""import os,hashlib,platform,subprocess,time
from pathlib import Path
started=time.monotonic()
repo=Path('/kaggle/working/hala-h47-source')
output=Path('/kaggle/working/hala-h47-paired-model-result')
os.environ['HALA_H47_NONCOMMERCIAL_FREE_RESEARCH']=__OPTIN__
if os.environ['HALA_H47_NONCOMMERCIAL_FREE_RESEARCH']!='1':
    raise RuntimeError('H47 explicit zero-cost noncommercial research opt-in required')
if platform.system()!='Linux' or not Path('/kaggle/working').is_dir() or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch':
    raise RuntimeError('Only private free Kaggle Batch supported')
if repo.exists() or output.exists():raise FileExistsError('No overwriting original run')
probe=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=20)
if probe.returncode!=0 or 'T4' not in probe.stdout:
    raise RuntimeError('No free T4 actually allocated')
os.environ.update(HF_HOME='/tmp/hala-h34-cache',HF_HUB_DISABLE_TELEMETRY='1',
    HF_HUB_DISABLE_IMPLICIT_TOKEN='1',DO_NOT_TRACK='1',
    WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
subprocess.run(['git','clone','--quiet','--depth','1','--single-branch','--branch',
                __BRANCH__,'https://github.com/TheHalfMoon/Hala.git',str(repo)],
                timeout=240,check=True)
def git(*cmd):
    return subprocess.check_output(['git',*cmd],cwd=repo,text=True,timeout=45).strip()
if git('rev-parse','HEAD')!=__COMMIT__ or git('status','--porcelain','--untracked-files=all'):
    raise RuntimeError('H47 signed source mismatched or dirty')
for name,digest in __FILES__.items():
    if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Signed H47 file SHA mismatch: '+name)
print('H47_SOURCE_PINNED_AND_FREE_T4_CONFIRMED',__COMMIT__,len(__FILES__),flush=True)
"""
    for a,b in {"__OPTIN__":repr("1" if research_allowed else "0"),
                "__BRANCH__":repr(SOURCE_BRANCH),
                "__COMMIT__":repr(commit),
                "__FILES__":repr(source)}.items():
        init=init.replace(a,b)
    run="""import os,signal,subprocess,sys,time,json
from pathlib import Path
proc=subprocess.Popen([sys.executable,'-u','-m','h47.experiment','--output',
    str(output),'--source-commit',__COMMIT__],cwd=repo,start_new_session=True)
try:
    rc=proc.wait(timeout=1130)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=15)
    raise RuntimeError('H47 external GPU Batch supervisor bounded timeout')
report={'schema':'hala.h47.parent.v1','source_commit':__COMMIT__,
        'worker_exit':rc,'model_qualified':False,'benchmark':False,
        'elapsed_s':time.monotonic()-started}
Path('/kaggle/working/h47-parent-exit.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\\n')
print('H47_WORKER_RETURN',rc,'EXTERNAL_REPLAY_REQUIRED',flush=True)
if rc!=0:raise RuntimeError('Retain original incomplete run; no automatic resubmit')
""".replace("__COMMIT__",repr(commit))
    notebook={"nbformat":4,"nbformat_minor":5,
              "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                          "protocol":PROTOCOL,"source_commit":commit,
                          "prompt_hashes":PROMPT_SHA,
                          "model":MODEL,"revision":REVISION,"cases":8,"attempts":16,
                          "research_optin":research_allowed,"private":True,
                          "model_qualified":False,"independent_test":False},
              "cells":[cell("markdown","# H47 same-T4 paired no-numeric-coordinate-example DEV experiment\n"
                             "Private free/noncommercial research only. 8 authored public DEV screenshots, "
                             "16 balanced prompt trials. No native actions or external benchmark.\n"),
                       cell("code",init),cell("code",run)]}
    file="hala-h47-paired-original-vs-no-number.ipynb"
    (target/file).write_text(json.dumps(notebook,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    meta={"id":KAGGLE_ID,"title":"hala h47 paired prompt dev eight","code_file":file,
          "language":"python","kernel_type":"notebook","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
          "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (target/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"source_sha":commit,"private":True,"cases":8,"paired_attempts":16,
            "research_allowed":research_allowed,"submitted":False,
            "files":len(source)}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-commit",required=True)
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--research-allowed",action="store_true")
    a=ap.parse_args()
    print(json.dumps(generate(a.root,a.source_commit,a.output,research_allowed=a.research_allowed),sort_keys=True))

"""Create-only opt-in private Kaggle notebook for H37 browser-raster baseline.

Never submits, trains, bills, changes project source, or modifies older evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h34.runner import MODEL,REVISION
from h37.runner import PROTOCOL,PROMPT_SHA,SOURCE_BRANCH

ID="abdulazizshehri/hala-h37-browser-grounding-dev-20261020"
FILES=tuple(
    ["h34/__init__.py","h34/runner.py","h34/protocol.py","h34/geometry.py",
     "h34/corpus.py","h34/render.py","h34/audit.py","h34/font_pin.json",
     "h36/__init__.py","h36/protocol.py",
     "h37/__init__.py","h37/scenes.py","h37/freeze.py","h37/runner.py",
     "h37/audit.py","tools/build_h37_kaggle.py","tools/build_h37_scenes.py",
     "tools/h37_capture.cjs"]
    + sorted("h37/assets/"+str(path.relative_to("h37/assets")).replace("\\","/")
             for path in Path("h37/assets").rglob("*") if path.is_file())
)


def cell(kind:str,text:str)->dict:
    data={"cell_type":kind,"metadata":{},"source":text.splitlines(keepends=True)}
    if kind=="code":data.update(execution_count=None,outputs=[])
    return data


def inventory(root:Path,sha:str)->dict:
    if not re.fullmatch("[a-f0-9]{40}",sha):
        raise ValueError("exact H37 signed source commit SHA required")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()!=sha:
        raise ValueError("git source head differs")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=root,text=True).strip():
        raise ValueError("build H37 notebook from a clean immutable source worktree")
    result={}
    for f in FILES:
        payload=(root/f).read_bytes()
        if payload!=subprocess.check_output(["git","show",f"{sha}:{f}"],cwd=root):
            raise ValueError("H37 source blob differs from Git: "+f)
        result[f]=hashlib.sha256(payload).hexdigest()
    return result


def generate(root:Path,source_commit:str,output:Path,*,research_allowed:bool=False)->dict:
    root,output=Path(root),Path(output)
    hashes=inventory(root,source_commit)
    if output.exists():raise FileExistsError("H37 notebook output must be new")
    output.mkdir(parents=True,exist_ok=False)
    bootstrap="""import hashlib,json,os,platform,subprocess,time
from pathlib import Path
start=time.monotonic()
source_dir=Path('/kaggle/working/hala-h37-source')
target=Path('/kaggle/working/hala-h37-baseline-output')
os.environ['HALA_H37_RESEARCH_ALLOWED']=__RESEARCH_OPT__
if os.environ['HALA_H37_RESEARCH_ALLOWED']!='1':
    raise RuntimeError('explicit private noncommercial research opt-in required')
if platform.system()!='Linux' or not Path('/kaggle/working').is_dir() or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch':
    raise RuntimeError('H37 requires Kaggle Batch Linux')
if source_dir.exists() or target.exists():
    raise FileExistsError('refuse to clobber Kaggle prior source/evidence')
os.environ.update(HF_HOME='/tmp/hala-h34-cache', HF_HUB_DISABLE_TELEMETRY='1',
    HF_HUB_DISABLE_IMPLICIT_TOKEN='1',DO_NOT_TRACK='1',WANDB_DISABLED='true',
    TOKENIZERS_PARALLELISM='false')
subprocess.run(['git','clone','--quiet','--depth','1','--single-branch',
    '--branch',__BRANCH__,'https://github.com/TheHalfMoon/Hala.git',str(source_dir)],
    timeout=240,check=True)
def git(*args):
    return subprocess.check_output(['git',*args],cwd=source_dir,text=True,timeout=40).strip()
if git('rev-parse','HEAD')!=__SOURCE_SHA__:
    raise RuntimeError('exact H37 source commit changed')
if git('status','--porcelain','--untracked-files=all'):
    raise RuntimeError('H37 cloned source is dirty')
sources=json.loads(__HASHES__)
for file,sha in sources.items():
    if hashlib.sha256((source_dir/file).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('H37 source blob changed: '+file)
print('H37_SOURCE_CONFIRMED',__SOURCE_SHA__,'FILES',len(sources))
"""
    for token,value in {
        "__RESEARCH_OPT__":repr("1" if research_allowed else "0"),
        "__BRANCH__":repr(SOURCE_BRANCH),
        "__SOURCE_SHA__":repr(source_commit),
        "__HASHES__":repr(json.dumps(hashes,sort_keys=True)),
    }.items():
        bootstrap=bootstrap.replace(token,value)
    launch="""import os,signal,subprocess,sys,time
proc=subprocess.Popen([sys.executable,'-u','-m','h37.runner','--output',
   '/kaggle/working/hala-h37-baseline-output','--source-commit',__SRC__],
   cwd=source_dir,start_new_session=True)
try:
    result=proc.wait(timeout=max(1,min(1250,1450-(time.monotonic()-start))))
except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=10)
    raise RuntimeError('H37 outer deadline exceeded; retain partial raw only')
if result!=0:
    raise RuntimeError('H37 worker failure; do not qualify model')
print('H37_NOTEBOOK_FINISHED_UNQUALIFIED; VERIFY_RAW_EXTERNALLY')
""".replace("__SRC__",repr(source_commit))
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "hala_protocol":PROTOCOL,"hala_source_commit":source_commit,
        "hala_prompt_sha256":PROMPT_SHA,
        "hala_model":MODEL,"hala_model_revision":REVISION,
        "hala_research_opt_in":research_allowed,"hala_submitted":False,
        "hala_model_qualified":False,
        "hala_blind_test":False,
        "hala_source_inventory_sha256":hashlib.sha256(
            json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    },"cells":[
        cell("markdown","# Hala H37 browser-rendered synthetic UI model baseline\n\n"
             "64 new authored offline Chromium browser PNGs, 32 DEV and 32 disjoint-"
             "domain TEST, Arabic/English. Not real external apps or blinded heldout "
             "evaluation. Private, zero-cost-only research. No UI actions.\n"),
        cell("code",bootstrap),
        cell("code",launch),
    ]}
    filename="hala-h37-browser-grounding-dev.ipynb"
    (output/filename).write_text(json.dumps(nb,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    meta={"id":ID,"title":ID.split("/")[1],"code_file":filename,
          "language":"python","kernel_type":"notebook","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
          "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (output/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"source_commit":source_commit,"files":len(FILES),"private":True,
            "research_allowed":research_allowed,"submitted":False,
            "output":str(output)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--research-allowed",action="store_true")
    args=parser.parse_args()
    print(json.dumps(generate(Path(args.root),args.source_commit,Path(args.output),
                              research_allowed=args.research_allowed),sort_keys=True))

if __name__=="__main__":
    main()

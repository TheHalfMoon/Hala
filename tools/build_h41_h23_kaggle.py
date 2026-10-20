"""Generate an opt-in private Kaggle notebook for the ORIGINAL SHA-frozen H23.

The historical source is never rewritten or refrozen. This is research only,
not a hosted service, production model, commercial or paid evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

H23_SHA = "f104e3e68fe378faf65c63248cb239bd2d171c28"
FREEZE_SHA = "4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a"
NOTEBOOK_ID = "abdulazizshehri/hala-h41-original-h23-cuda-lifecycle-20261020"
FONT_SHA = "ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280"
H41_BRANCH = "ps/h41-h23-real-cuda-eval"


def source_check(repo: Path) -> dict:
    repo=Path(repo)
    def git(*args:str)->bytes:
        return subprocess.check_output(["git",*args],cwd=repo,timeout=40)
    if not re.fullmatch("[0-9a-f]{40}",H23_SHA):
        raise ValueError("pin must be exact SHA40")
    freeze=git("show",f"{H23_SHA}:experiments/serving-lifecycle/source-freeze.json")
    if hashlib.sha256(freeze).hexdigest()!=FREEZE_SHA:
        raise ValueError("original frozen H23 artifact changed")
    obj=json.loads(freeze)
    if len(obj.get("files",{}))!=53:
        raise ValueError("expected original 53 files")
    for name,digest in obj["files"].items():
        if (not name.endswith((".py",".json")) or
                hashlib.sha256(git("show",f"{H23_SHA}:{name}")).hexdigest()!=digest):
            raise ValueError("original H23 frozen source mismatch: "+name)
    if git("merge-base","--is-ancestor",H23_SHA,git("rev-parse","HEAD").decode().strip())!=b"":
        raise ValueError("invalid source ancestor proof")
    return {"historic_source_sha":H23_SHA,"source_freeze_sha256":FREEZE_SHA,
            "frozen_source_files":53}


def _cell(t:str,source:str)->dict:
    c={"cell_type":t,"metadata":{},"source":source.splitlines(keepends=True)}
    if t=="code":c.update(execution_count=None,outputs=[])
    return c


PREPARE = """import os,sys,platform,subprocess,hashlib,json,time,pathlib,importlib.metadata
from pathlib import Path
start=time.monotonic()
working=Path('/kaggle/working')
historical=working/'hala-h41-h23-original-source'
run_path=working/'hala-h41-h23-run'
receipt=working/'h41-h23-launch.json'
if (platform.system()!='Linux' or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch'
    or not working.is_dir() or os.environ.get('HALA_REMOTE_DEVELOPMENT_ALLOWED')=='1'):
    raise RuntimeError('H41 requires actual Kaggle Batch Linux, no inherited opt-in')
if historical.exists() or run_path.exists() or receipt.exists():
    raise FileExistsError('H41 refuses overwrite or resume')
if subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=18).returncode!=0:
    raise RuntimeError('H41 requires a real free Kaggle GPU allocation')
os.environ.update(HALA_REMOTE_DEVELOPMENT_ALLOWED='1',
 HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
 DO_NOT_TRACK='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
# Read-only historical Git checkout: never patch source, freeze, permissions or model.
subprocess.run(['git','clone','--quiet','--filter=blob:none','--no-checkout',
                '--depth','1','--single-branch','--branch',__BRANCH__,
                'https://github.com/TheHalfMoon/Hala.git',str(historical)],
                timeout=240,check=True)
subprocess.run(['git','fetch','--quiet','--depth','1','origin',__HISTORIC__],
               cwd=historical,timeout=220,check=True)
subprocess.run(['git','checkout','--quiet','--detach',__HISTORIC__],
               cwd=historical,timeout=90,check=True)
def git(*args):
    return subprocess.check_output(['git',*args],cwd=historical,text=True,timeout=30).strip()
if git('rev-parse','HEAD')!=__HISTORIC__:
    raise RuntimeError('historical commit mismatch')
if git('status','--porcelain','--untracked-files=all'):
    raise RuntimeError('historical source is not clean')
freeze=(historical/'experiments/serving-lifecycle/source-freeze.json').read_bytes()
if hashlib.sha256(freeze).hexdigest()!=__FREEZE__:
    raise RuntimeError('original H23 source freeze changed')
# Admission checks original frozen source files, BEFORE installing dependencies.
sys.path.insert(0,str(historical))
from evaluation.serving_lifecycle import verify_sources
verify_sources(__FREEZE__)
# Only exact prescribed Pillow version is installed in the isolated ephemeral Kaggle VM.
if importlib.metadata.version('Pillow')!='11.3.0':
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                    '--no-input','--no-deps','--quiet','Pillow==11.3.0'],
                   timeout=180,check=True)
import PIL
from PIL import features
font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
pin_sha=__FONT_SHA__
if not font.is_file() or hashlib.sha256(font.read_bytes()).hexdigest()!=pin_sha:
    raise RuntimeError('original H23 font not matched; fail closed')
import torch,transformers
if (importlib.metadata.version('Pillow')!='11.3.0'
    or not features.check_feature('raqm')
    or torch.__version__.split('+')[0]!='2.11.0'
    or transformers.__version__!='5.16.1'
    or not torch.cuda.is_available()):
    raise RuntimeError('original exact H23 runtime pins or RAQM/CUDA not met')
from evaluation.serving_lifecycle import preflight
preflight('kaggle',str(font),__FREEZE__,__HISTORIC__)
info={'schema':'hala.h41.h23_original_runtime.v1',
 'original_source_commit':git('rev-parse','HEAD'),
 'original_freeze_sha256':__FREEZE__,
 'historical_source_unmodified':True,
 'model_id':'Qwen/Qwen3-VL-2B-Instruct',
 'torch':torch.__version__,'transformers':transformers.__version__,
 'Pillow':importlib.metadata.version('Pillow'),
 'raqm':features.check_feature('raqm'),
 'font_sha256':hashlib.sha256(font.read_bytes()).hexdigest(),
 'cuda_devices':[str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())],
 'gpu_observed':subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=18).stdout,
 'model_qualified':False,'actual_cuda_cancel_qualified':False,
 'native_execution':False,'noncommercial_research_only':True}
receipt.write_text(json.dumps(info,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
print('H41_ORIGINAL_H23_HARD_GATES_PASS',__HISTORIC__,__FREEZE__,flush=True)
"""
LAUNCH = """import os,signal,subprocess,sys,json,time,hashlib
from pathlib import Path
target=Path('/kaggle/working/hala-h41-h23-run')
receipt=Path('/kaggle/working/h41-h23-launch.json')
info=json.loads(receipt.read_text('utf-8'))
log=Path('/kaggle/working/h41-h23-operator.log')
cmd=[sys.executable,'-u','-m','evaluation.serving_lifecycle',
  '--provider','kaggle','--freeze-sha256',__FREEZE__,
  '--source-commit',__HISTORIC__,
  '--font','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
  '--output',str(target),'--faults']
with log.open('xb') as output:
    proc=subprocess.Popen(cmd,cwd=historical,stdout=output,
                          stderr=subprocess.STDOUT,start_new_session=True)
    timed_out=False
    try:
        code=proc.wait(timeout=1000)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(proc.pid,signal.SIGTERM)
        try:code=proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            code=proc.wait(timeout=12)
info.update(operator_exit_code=code,operator_timed_out=timed_out,
            elapsed_seconds=time.monotonic()-start,
            operator_stdout_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            operator_output_available=target.exists(),
            state='incomplete' if timed_out or code!=0 else 'completed_unqualified',
            model_qualified=False,actual_cuda_cancel_qualified=False,
            native_execution=False)
receipt.write_text(json.dumps(info,sort_keys=True,indent=2)+'\\n',encoding='utf-8')
print('H41_ORIGINAL_H23_OPERATOR_EXIT',code,'TIMEOUT',timed_out,flush=True)
print('H41_FINAL_RECEIPT_SHA256',hashlib.sha256(receipt.read_bytes()).hexdigest(),flush=True)
print('H41_NO_QUALIFICATION_BEFORE_INDEPENDENT_EVIDENCE_REPLAY',flush=True)
if timed_out or code!=0:raise RuntimeError('Original H23 remote lifecycle incomplete; retain raw evidence')
"""


def make_notebook(repo:Path,out:Path,research_opt_in:bool=False)->dict:
    meta=source_check(repo)
    out=Path(out)
    if out.exists():
        raise FileExistsError("Never overwrite an H41 private Kaggle bundle")
    out.mkdir(parents=True,exist_ok=False)
    prep=PREPARE
    launch=LAUNCH
    for token,value in {
        "__BRANCH__":repr(H41_BRANCH),
        "__HISTORIC__":repr(H23_SHA),
        "__FREEZE__":repr(FREEZE_SHA),
        "__FONT_SHA__":repr(FONT_SHA),
    }.items():
        prep=prep.replace(token,value)
        launch=launch.replace(token,value)
    if not research_opt_in:
        prep="raise RuntimeError('Explicit zero-cost research opt-in required')\n"+prep
    notebook={
        "nbformat":4,"nbformat_minor":5,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "hala_protocol":"h41_original_h23_cuda_private",
                    "hala_original_source_sha":H23_SHA,"hala_original_freeze_sha256":FREEZE_SHA,
                    "hala_research_opt_in":bool(research_opt_in),
                    "hala_model_qualified":False,"hala_native_execution":False,
                    "hala_h23_status":"unqualified_until_independent_review"},
        "cells":[
            _cell("markdown","# Hala H41 private original H23 CUDA lifecycle research\n"
                  "Unmodified historically reviewed 53-file source at exact SHA. "
                  "Private Kaggle Batch free GPU only. No native OS or commercial uses. "
                  "A completed notebook is NOT CUDA cancellation qualification.\n"),
            _cell("code",prep),
            _cell("code",launch)
        ]
    }
    nb="hala-h41-original-h23-cuda.ipynb"
    (out/nb).write_text(json.dumps(notebook,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    kernel={"id":NOTEBOOK_ID,"title":"Hala H41 Original H23 CUDA Lifecycle",
            "code_file":nb,"language":"python","kernel_type":"notebook",
            "is_private":True,"enable_gpu":True,"enable_tpu":False,
            "enable_internet":True,"dataset_sources":[],"competition_sources":[],
            "kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(kernel,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {**meta,"notebook_path":str(out),"private":True,"submitted":False,
            "opt_in":research_opt_in}


if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--repo",type=Path,default=Path("."))
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--research-opt-in",action="store_true")
    args=a.parse_args()
    print(json.dumps(make_notebook(args.repo,args.output,args.research_opt_in),sort_keys=True))

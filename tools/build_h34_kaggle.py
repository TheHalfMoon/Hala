"""Generate a private H34 Kaggle notebook; never submits or installs packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

FILES = ("h34/__init__.py", "h34/corpus.py", "h34/geometry.py", "h34/protocol.py",
         "h34/render.py", "h34/runner.py", "h34/audit.py", "h34/font_pin.json",
         "h34/DEJAVU_LICENSE.txt", "h34/frozen/cases.json", "tools/build_h34_kaggle.py")
BRANCH = "ps/h34-bilingual-visual-dev"
KERNEL_ID = "abdulazizshehri/hala-h34-bilingual-visual-dev-20261019"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def inventory(root: Path, source_commit: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", source_commit) or _git(root, "rev-parse", "HEAD") != source_commit:
        raise ValueError("exact local source HEAD SHA required")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source must be clean and immutable before bundle generation")
    files = {}
    for name in FILES:
        payload = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=root)
        files[name] = hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "sha256": digest}


def cell(kind: str, source: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def generate(root: Path, source_commit: str, output: Path, *, research_allowed: bool = False) -> dict:
    root, output = Path(root), Path(output)
    inv = inventory(root, source_commit)
    if output.exists():
        raise FileExistsError("H34 bundle output cannot overwrite prior artifacts")
    output.mkdir(parents=True, exist_ok=False)
    verify = """import hashlib, json, subprocess, platform, time
from pathlib import Path
H34_BOOTSTRAP_STARTED=time.monotonic()
root=Path('/kaggle/working/hala-h34-research-src')
os.environ['HALA_H34_RESEARCH_ALLOWED']=RESEARCH_OPT_IN
if os.environ.get('HALA_H34_RESEARCH_ALLOWED')!='1': raise RuntimeError('explicit private research opt-in required')
if platform.system()!='Linux' or not Path('/kaggle/working').is_dir(): raise RuntimeError('Kaggle Linux only')
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch': raise RuntimeError('retained Kaggle Batch only')
for name in ('hala-h34-research-src','hala-h34-dev-out'):
    if Path('/kaggle/working',name).exists(): raise FileExistsError('H34 paths are exclusive; no resume')
os.environ['HF_HOME']='/tmp/hala-h34-cache'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['DO_NOT_TRACK']='1'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
subprocess.run(['git','clone','--quiet','--depth','1','--single-branch','--branch',BRANCH,REPO,str(root)],check=True,timeout=300)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=20).strip()
if actual != SOURCE_SHA: raise RuntimeError('H34 immutable source commit mismatch')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=root,text=True,timeout=20).strip(): raise RuntimeError('dirty remote source')
files=json.loads(INVENTORY_JSON)
for name,expected in files.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise RuntimeError('source content inventory mismatch: '+name)
print('H34_IMMUTABLE_SOURCE',actual,'CONTENT_INVENTORY',INVENTORY_SHA)
"""
    verify = ("import os\n" + verify.replace("BRANCH", repr(BRANCH)).replace("REPO", repr("https://github.com/TheHalfMoon/Hala.git"))
              .replace("SOURCE_SHA", repr(source_commit)).replace("INVENTORY_JSON", repr(json.dumps(inv["files"], sort_keys=True)))
              .replace("INVENTORY_SHA", repr(inv["sha256"]))
              .replace("RESEARCH_OPT_IN", repr("1" if research_allowed else "0")))
    deps = """import importlib.metadata
required={'torch':'2.11.0+cu128','transformers':'5.16.1','accelerate':'1.14.0','safetensors':'0.8.0','Pillow':'12.3.0'}
print('H34_REQUIRED_PACKAGES',json.dumps(required,sort_keys=True))
# The owned runner records observations and fails closed before any model load.
"""
    launch = f"""import subprocess,sys,signal
process=subprocess.Popen([sys.executable,'-u','-m','h34.runner','--output','/kaggle/working/hala-h34-dev-out','--source-commit','{source_commit}'],cwd='/kaggle/working/hala-h34-research-src',start_new_session=True)
try:
    code=process.wait(timeout=max(1,min(1500,1770-(time.monotonic()-H34_BOOTSTRAP_STARTED))))
except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)
    raise RuntimeError('H34 outer safety deadline; partial evidence only')
if code: raise RuntimeError('H34 supervisor failed; partial evidence only')
print('H34_ATTEMPT_RAW_OUTPUT_REQUIRES_EXTERNAL_SHA_AUDIT; NO_MODEL_QUALIFICATION')
"""
    notebook = {"nbformat":4,"nbformat_minor":5,
                "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                            "hala_source_commit":source_commit,"hala_case_sha256":inv["files"]["h34/frozen/cases.json"],
                            "hala_content_inventory_sha256":inv["sha256"],"hala_model": "Qwen/Qwen3-VL-2B-Instruct",
                            "hala_model_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
                            "hala_split":"prospective synthetic DEV", "hala_submitted":False,
                            "hala_research_opt_in": research_allowed, "hala_resume":False,
                            "hala_inference_run":False, "hala_model_qualified":False},
                "cells":[cell("markdown", "# Hala H34 private synthetic bilingual pixel-grounding DEV\n\n"
                        "32 fixed public DEV cases, 16 English and 16 Arabic. This is synthetic rendered GUI evidence, not real captured UI or blind TEST. No training, action execution, API invocation, or production qualification. Actual rendering requires exact pinned font, Linux RAQM and Pillow. Arabic wording has not received independent human linguistic adjudication.\n"),
                          cell("code", verify), cell("code", deps), cell("code", launch)]}
    name = "hala-h34-bilingual-visual-dev.ipynb"
    (output/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    kernel = {"id":KERNEL_ID,"title":KERNEL_ID.split('/')[1],"code_file":name,
              "language":"python","kernel_type":"notebook","is_private":True,
              "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
              "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (output/"kernel-metadata.json").write_text(json.dumps(kernel,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"output":str(output),"source_commit":source_commit,"inventory_sha256":inv["sha256"],"submitted":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--research-allowed", action="store_true", help="Explicit authorization for this private research bundle")
    args=parser.parse_args()
    print(json.dumps(generate(Path(args.root),args.source_commit,Path(args.output),research_allowed=args.research_allowed),sort_keys=True))


if __name__=="__main__":
    main()

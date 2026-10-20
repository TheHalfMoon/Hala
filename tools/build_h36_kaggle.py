"""Opt-in, immutable, private Kaggle H36 *bundle builder* (never submits).

Requires a clean exact Git source SHA, no credentials, paid compute or network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h34.runner import MODEL, REVISION
from h36.protocol import PROTOCOL, plan_sha, prompt_sha, ARMS
from tools.build_h34_kaggle import FILES as H34_FILES

FILES = tuple(dict.fromkeys((*H34_FILES, "h36/__init__.py", "h36/protocol.py",
                             "h36/runner.py", "h36/audit.py", "tools/build_h36_kaggle.py")))
BRANCH = "ps/h36-paired-prompt-ablation"
KERNEL_ID = "abdulazizshehri/hala-h36-paired-prompt-dev-20261020"


def inventory(root: Path, source_commit: str) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("exact frozen source commit required")
    root = Path(root)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != source_commit:
        raise ValueError("source HEAD differs from specified SHA")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                               cwd=root, text=True).strip():
        raise ValueError("source must be clean")
    source = {}
    for name in FILES:
        data = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=root)
        source[name] = hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256(json.dumps(source, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"files": source, "sha256": digest}


def cell(kind: str, content: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def generate(root: Path, source_commit: str, output: Path,
             *, research_allowed: bool = False) -> dict:
    info = inventory(root, source_commit)
    output = Path(output)
    if output.exists():
        raise FileExistsError("H36 create-only output; cannot overwrite")
    output.mkdir(parents=True, exist_ok=False)
    bootstrap = """import hashlib, json, os, platform, subprocess, time
from pathlib import Path
H36_BOOTSTRAP_START = time.monotonic()
source_dir = Path('/kaggle/working/hala-h36-immutable-src')
output_dir = Path('/kaggle/working/hala-h36-paired-output')
os.environ['HALA_H36_RESEARCH_ALLOWED'] = RESEARCH_OPT_IN
if os.environ['HALA_H36_RESEARCH_ALLOWED'] != '1':
    raise RuntimeError('explicit H36 noncommercial research opt-in required')
if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
    raise RuntimeError('only Kaggle Linux allowed')
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch':
    raise RuntimeError('retained Batch is required')
if source_dir.exists() or output_dir.exists():
    raise FileExistsError('cannot clobber previous H36 source/evidence')
os.environ.update(HF_HOME='/tmp/hala-h34-cache', HF_HUB_DISABLE_TELEMETRY='1',
                  HF_HUB_DISABLE_IMPLICIT_TOKEN='1', DO_NOT_TRACK='1',
                  WANDB_DISABLED='true', TOKENIZERS_PARALLELISM='false')
subprocess.run(['git', 'clone', '--quiet', '--depth', '1', '--single-branch',
                '--branch', BRANCH, 'https://github.com/TheHalfMoon/Hala.git',
                str(source_dir)], check=True, timeout=300)
def git(*args):
    return subprocess.check_output(['git', *args], cwd=source_dir, text=True, timeout=25).strip()
if git('rev-parse', 'HEAD') != SOURCE_SHA:
    raise RuntimeError('source commit changed; abort run')
if git('status', '--porcelain', '--untracked-files=all'):
    raise RuntimeError('remote source is dirty')
source_manifest = json.loads(FILE_MANIFEST)
for name, expected in source_manifest.items():
    if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('signed source file mismatch: ' + name)
print('H36_SHA_BOUND_SOURCE', SOURCE_SHA, 'FILES_SHA256', __INVENTORY_DIGEST__)
"""
    for old, replacement in (("RESEARCH_OPT_IN", repr("1" if research_allowed else "0")),
                             ("BRANCH", repr(BRANCH)), ("SOURCE_SHA", repr(source_commit)),
                             ("FILE_MANIFEST", repr(json.dumps(info["files"], sort_keys=True))),
                             ("__INVENTORY_DIGEST__", repr(info["sha256"]))):
        bootstrap = bootstrap.replace(old, replacement)
    # The control comparison must run in one process/model and same VM.
    launch = """import os, signal, subprocess, sys, time
cmd = [sys.executable, '-u', '-m', 'h36.runner',
       '--output', '/kaggle/working/hala-h36-paired-output',
       '--source-commit', SOURCE_SHA]
proc = subprocess.Popen(cmd, cwd=source_dir, start_new_session=True)
try:
    code = proc.wait(timeout=max(1, min(1500, 1740-(time.monotonic()-H36_BOOTSTRAP_START))))
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    raise RuntimeError('outer H36 bounded deadline exceeded; retain partial output only')
if code != 0:
    raise RuntimeError('H36 owned process failed; no qualification')
print('H36_REAL_RAW_REQUIRES_SEPARATE_SHA_BOUND_AUDIT')
""".replace("SOURCE_SHA", repr(source_commit))
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "hala_source_commit": source_commit,
                             "hala_source_files_sha256": info["sha256"],
                             "hala_protocol": PROTOCOL, "hala_plan_sha256": plan_sha(),
                             "hala_prompt_sha256": {arm: prompt_sha(arm) for arm in ARMS},
                             "hala_model": MODEL, "hala_model_revision": REVISION,
                             "hala_split": "exposed H34 synthetic DEV; NOT unseen",
                             "hala_research_opt_in": research_allowed,
                             "hala_submitted": False, "hala_model_qualified": False},
                "cells": [cell("markdown", "# Hala H36 paired no-numeric-example prompt ablation\n\n"
                               "32 existing exposed first-party H34 synthetic DEV images; two matched prompt arms, "
                               "64 inferences in one Kaggle T4 Batch VM. Same model, frames and evaluation, "
                               "alternating arm order. **No independent held-out TEST or real GUI qualification.**"
                               " No user or device actions. Zero-cost, noncommercial research only.\n"),
                          cell("code", bootstrap),
                          cell("code", "import importlib.metadata\nprint('H36_PINNED_DEPS', {name: importlib.metadata.version(name) for name in ('torch','transformers','accelerate','safetensors','Pillow')})\n"),
                          cell("code", launch)]}
    name = "hala-h36-paired-prompt-dev.ipynb"
    (output / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": KERNEL_ID, "title": KERNEL_ID.split("/")[1], "code_file": name,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return {"source_commit": source_commit, "inventory_sha256": info["sha256"],
            "plan_sha256": plan_sha(), "submitted": False, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--research-allowed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(Path(args.root), args.source_commit, Path(args.output),
                              research_allowed=args.research_allowed), sort_keys=True))


if __name__ == "__main__":
    main()

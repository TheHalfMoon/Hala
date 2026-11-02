"""H353: reproducible distinct owner-private counterbalanced Qwen experiment.

Frozen validated H352 source is embedded by Git object identity, not copied
from mutable filesystem state. One Kaggle code script + private metadata;
NO submission, images, gold, model outputs or weights during build.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import tempfile

from h352.build_private_payload import build as h352_build, SCRIPT as H352_SCRIPT

SOURCE_REV = "37a7224f4886618e44788de92ce48ed0df2114f3"
KERNEL_ID = "abdulazizshehri/hala-h353-order-balanced-pixel-qwen-t4"
TITLE = "Hala H353 Order Balanced Pixel Qwen T4"
SCRIPT = "h353_owner_private_order_balanced_qwen.py"
STATUS = "H353_PRIVATE_ORDER_BALANCED_REAL_QWEN_EXPOSED_PIXEL_SNAP_PAIRED"
REPORT = "hala-h353-private-order-balanced-paired.json"
DIAG = "hala-h353-private-stage-only.json"
FLAG = "HALA_H353_PRIVATE_NONCOMMERCIAL_FREE_T4"
_REPLACE = {
    "HALA_H352_PRIVATE_NONCOMMERCIAL_FREE_T4": FLAG,
    "hala-h352-private-paired-snap-aggregate.json": REPORT,
    "hala-h352-private-stage-only.json": DIAG,
    "H352_PRIVATE_REAL_QWEN_EXPOSED_PIXEL_SNAP_PAIRED": STATUS,
    "H352_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE":
        "H353_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE",
}

def need(ok):
    if not ok:
        raise ValueError("H353 EXACT FROZEN PRIVATE SOURCE REFUSED")

def _pinned_file(repo: Path, relative: str) -> str:
    raw = subprocess.run(
        ["git", "show", SOURCE_REV + ":" + relative], cwd=repo,
        capture_output=True, check=True, timeout=20).stdout
    need(0 < len(raw) < 45000 and b"\x00" not in raw)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("H353 SOURCE ENCODING REFUSED") from None
    return value

def build(repo: Path, destination: Path) -> dict:
    need(isinstance(repo, Path) and isinstance(destination, Path)
         and repo.is_dir() and not destination.exists()
         and ".git" not in destination.parts
         and not destination.is_relative_to(repo))
    source = _pinned_file(repo, "h352/private_qwen_paired_snap.py")
    pixel = _pinned_file(repo, "h352/pixel_button_snap.py")
    # Check exact intended order-counterbalanced protocol is in frozen source.
    for term in ("def paired_arm_order(scene):",
                 'candidate_first = bool((index % 2) ^ (scene[1] == "ar"))',
                 "def native_action_changed(",
                 'need(row["changed"] or row["baseline"] == row["candidate"])'):
        need(term in source)
    for term in ("def pixel_button_rectangles(", "def snap_native_first_click("):
        need(term in pixel)
    with tempfile.TemporaryDirectory() as temporary:
        src = Path(temporary) / "h352"
        src.mkdir()
        (src / "private_qwen_paired_snap.py").write_text(
            source, encoding="utf-8", newline="\n")
        (src / "pixel_button_snap.py").write_text(
            pixel, encoding="utf-8", newline="\n")
        base = Path(temporary) / "original-code-only"
        h352_build(src / "private_qwen_paired_snap.py",
                   src / "pixel_button_snap.py", base)
        compiled = (base / H352_SCRIPT).read_text(encoding="utf-8")
    for previous, replacement in _REPLACE.items():
        need(previous in compiled)
        compiled = compiled.replace(previous, replacement)
    need(all(previous not in compiled for previous in _REPLACE))
    compile(compiled, SCRIPT, "exec")
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / SCRIPT
    payload.write_text(compiled, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {
        "status": "H353_FROZEN_COUNTERBALANCED_PRIVATE_CODE_ONLY",
        "source_rev": SOURCE_REV,
        "code_sha256": sha256(payload.read_bytes()).hexdigest(),
        "private": True, "gpu_job_submitted": False,
        "media_or_weights_uploaded": 0, "file_count": 2,
    }

"""H354 fresh private owner T4 experiment: conservative click preservation.

H353 frozen order-balanced experiment is reused, including no-op fail-closed
guard and public H319 source/Qwen revisions. Only the embedded screenshot-
pixel snap policy is replaced with conservative inside-button preservation.
Does NOT upload images, model predictions, labels, datasets or weights.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import tempfile

from h353.build_private_payload import (
    build as h353_build, _pinned_file,
    SOURCE_REV, SCRIPT as PREV_SCRIPT,
)
from h354.preserve_valid_clicks import FROZEN_POLICY_SUFFIX

KERNEL_ID = "abdulazizshehri/hala-h354-preserve-valid-pixel-qwen-t4"
TITLE = "Hala H354 Preserve Valid Pixel Qwen T4"
SCRIPT = "h354_private_conservative_qwen.py"
FLAG = "HALA_H354_PRIVATE_NONCOMMERCIAL_FREE_T4"
REPORT = "hala-h354-private-conservative-paired.json"
DIAG = "hala-h354-private-stage-only.json"
STATUS = "H354_PRIVATE_CONSERVATIVE_ORDER_BALANCED_REAL_QWEN"
PREV_STATUS = "H353_PRIVATE_ORDER_BALANCED_REAL_QWEN_EXPOSED_PIXEL_SNAP_PAIRED"
SOURCE_SHA = SOURCE_REV


def need(ok):
    if not ok:
        raise ValueError("H354 FROZEN PRIVATE STUDY REFUSED")


def build(repo: Path, destination: Path):
    need(isinstance(repo, Path) and repo.is_dir()
         and isinstance(destination, Path) and not destination.exists()
         and ".git" not in destination.parts
         and not destination.is_relative_to(repo))
    source = _pinned_file(repo, "h352/pixel_button_snap.py")
    need(source.count("def snap_native_first_click(") == 1
         and "def pixel_button_rectangles(" in source)
    suffix = FROZEN_POLICY_SUFFIX
    need(suffix.count("def snap_native_first_click(") == 1)
    new_policy = source + suffix
    compile(new_policy, "h354/pixel_button_snap.py", "exec")
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp) / "h353"
        h353_build(repo, base)
        compiled = (base / PREV_SCRIPT).read_text(encoding="utf-8")
    old_policy_expression = "SNAP_CODE = " + repr(source)
    need(compiled.count(old_policy_expression) == 1)
    compiled = compiled.replace(
        old_policy_expression, "SNAP_CODE = " + repr(new_policy))
    replacements = {
        "HALA_H353_PRIVATE_NONCOMMERCIAL_FREE_T4": FLAG,
        "hala-h353-private-order-balanced-paired.json": REPORT,
        "hala-h353-private-stage-only.json": DIAG,
        PREV_STATUS: STATUS,
        "H353_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE":
            "H354_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE",
    }
    for original, replacement in replacements.items():
        need(original in compiled)
        compiled = compiled.replace(original, replacement)
    need(all(original not in compiled for original in replacements))
    compile(compiled, SCRIPT, "exec")
    metadata = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    file = destination / SCRIPT
    file.write_text(compiled, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {
        "status": "H354_FROZEN_CONSERVATIVE_CODE_ONLY_NOT_SUBMITTED",
        "parent_source_revision": SOURCE_SHA,
        "pixel_policy_sha256": sha256(new_policy.encode()).hexdigest(),
        "uploaded_code_sha256": sha256(file.read_bytes()).hexdigest(),
        "private": True, "gpu_submitted": False,
        "files": 2, "media_gold_weights": 0,
    }

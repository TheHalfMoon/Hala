"""Build one H352 PRIVATE free-T4 code-only paired real-Qwen payload.

Absolutely no Kaggle submission or screenshot/weight/gold data in build.
"""
from __future__ import annotations
import argparse
import json
import re
import textwrap
from pathlib import Path
from hashlib import sha256

from h282.build_private_payload import INSTALL_H282
from h352.private_qwen_paired_snap import FLAG, PIN

TITLE = "Hala H352 Paired Pixel Snap Qwen T4"
KERNEL_ID = "abdulazizshehri/hala-h352-paired-pixel-snap-qwen-t4"
SOURCE = "private_qwen_paired_snap.py"
SNAP = "pixel_button_snap.py"
SCRIPT = "h352_private_paired_qwen_snap.py"
PLACEHOLDER = 'SNAP_CODE = "H352_SNAP_SOURCE_PENDING"'


def build(source: Path, snap: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE or source.parent.name != "h352"
            or not isinstance(snap, Path) or not snap.is_file()
            or snap.name != SNAP or snap.parent != source.parent
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H352 only a fresh source-owned private code bundle")
    if KERNEL_ID != "abdulazizshehri/" + re.sub(
        r"[^a-z0-9]+", "-", TITLE.lower()).strip("-"):
        raise ValueError("H352 private kernel slug mismatch")
    raw = source.read_text(encoding="utf-8")
    pixel = snap.read_text(encoding="utf-8")
    header = "from __future__ import annotations\n"
    required = (
        'PIN = "' + PIN + '"',
        'FLAG = "' + FLAG + '"',
        PLACEHOLDER,
        "def verify_free(", "def pinned_source(",
        "def measure_one_paired(", "def aggregate_pairs(",
        "def safe_report(", "def run():",
        'if __name__ == "__main__":',
    )
    if (raw.count(header) != 1 or raw.count(PLACEHOLDER) != 1
            or not all(term in raw for term in required)
            or pixel.count(header) != 1
            or "def snap_native_first_click(" not in pixel
            or "def pixel_button_rectangles(" not in pixel):
        raise ValueError("H352 frozen private logic/pixel code mismatch")
    updated = raw.replace(PLACEHOLDER, "SNAP_CODE = " + repr(pixel))
    bootstrap = ("try:\n" + textwrap.indent(
        INSTALL_H282.replace("HALA_H282_PRIVATE_FREE_NONCOMMERCIAL", FLAG),
        "    ") + "\nexcept Exception:\n"
        + "    raise RuntimeError('H352 private dependencies unavailable') from None\n")
    compiled = header + bootstrap + updated.replace(header, "", 1)
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
    (destination / SCRIPT).write_text(compiled, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {
        "status": "H352_PRIVATE_PAIRED_CODE_ONLY_NOT_SUBMITTED",
        "uploaded_code_sha256": sha256(compiled.encode()).hexdigest(),
        "pixel_policy_sha256": sha256(pixel.encode()).hexdigest(),
        "private": True, "free_t4_requested": True,
        "uploaded_files": 2, "media_weights_gold_uploaded": 0,
        "gpu_job_submitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--snap", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.source, args.snap, args.destination)
    # Source hashes are public code provenance, not private model outputs.
    print(json.dumps(report, sort_keys=True))

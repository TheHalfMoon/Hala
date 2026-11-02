"""Create one fresh, private, code-only FREE T4 H351 Kaggle research payload.

Build does NOT submit a GPU job or copy screenshots, labels or weights.
"""
from __future__ import annotations
import argparse
import json
import re
import textwrap
from pathlib import Path

from h282.build_private_payload import INSTALL_H282
from h351.private_qwen_miss_distance import FLAG, PIN

TITLE = "Hala H351 Qwen Miss Distance T4"
KERNEL_ID = "abdulazizshehri/hala-h351-qwen-miss-distance-t4"
SOURCE = "private_qwen_miss_distance.py"
SCRIPT = "h351_private_qwen_miss_distance.py"

def build(source: Path, destination: Path):
    if (not isinstance(source, Path) or source.name != SOURCE
            or source.parent.name != "h351" or not source.is_file()
            or not isinstance(destination, Path) or destination.exists()
            or ".git" in destination.parts
            or destination.is_relative_to(source.parent)):
        raise ValueError("H351 fresh private code-only bundle required")
    slug = re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-")
    if KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H351 private kernel slug must match title")
    raw = source.read_text(encoding="utf-8")
    header = "from __future__ import annotations\n"
    necessary = (
        'PIN = "' + PIN + '"',
        'FLAG = "' + FLAG + '"',
        "def qualify_free(",
        "def original_pinned_source(",
        "def validate_observed_counts(",
        "def safe_report(",
        "def run():",
        'if __name__ == "__main__":',
    )
    if raw.count(header) != 1 or not all(x in raw for x in necessary):
        raise ValueError("H351 original source identity not verified")
    install = INSTALL_H282.replace(
        "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL", FLAG)
    bootstrap = (
        "try:\n" + textwrap.indent(install, "    ")
        + "\nexcept Exception:\n"
        + "    raise RuntimeError('H351 dependencies unavailable') from None\n"
    )
    compiled = header + bootstrap + raw.replace(header, "", 1)
    compile(compiled, SCRIPT, "exec")
    metadata = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT,
        "language": "python", "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT).write_text(compiled, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {
        "status": "H351_FRESH_PRIVATE_CODE_ONLY_NONCOMMERCIAL",
        "kernel": KERNEL_ID,
        "code_files": 1, "metadata_files": 1,
        "images_or_weights_uploaded": 0,
        "gpu_job_submitted": False,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination), sort_keys=True))

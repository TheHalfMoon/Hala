"""H324: build a create-only, code-only owner-private FREE T4 payload.

This builder does NOT submit or run Kaggle. Separate owner account privacy,
free quota, source safety and prospective distinct-study checks are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

from h282.build_private_payload import INSTALL_H282

TITLE = "Hala H324 Private Pretrained BeaconBoard T4"
KERNEL_ID = "abdulazizshehri/hala-h324-private-pretrained-beaconboard-t4"
SCRIPT = "h324_private_pretrained_beaconboard.py"
SOURCE = "private_pretrained_beaconboard_pilot.py"
INSTALL = INSTALL_H282.replace(
    "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL",
    "HALA_H324_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4",
)
MARKERS = (
    'HEAD = "651364ab952c7e90dabb4c7c240e312ebd8d417f"',
    'HTML_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"',
    'FLAG = "HALA_H324_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4"',
    'INDICES = (0, 6)',
    "def attest_free_runtime(",
    "def aggregate(",
    "def checked_checkout(",
    "def run() -> None:",
    'if __name__ == "__main__":',
)


def build(source: Path, dest: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE or source.parent.name != "h324"
            or not isinstance(dest, Path) or dest.exists()
            or dest.is_relative_to(source.parent)
            or ".git" in dest.parts):
        raise ValueError("H324 create-only code-only payload refused")
    body = source.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    if body.count(future) != 1 or not all(x in body for x in MARKERS):
        raise ValueError("H324 frozen source contract drift")
    slug = re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-")
    if KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H324 owner private kernel slug mismatch")
    bootstrap = (future + "try:\n" + textwrap.indent(INSTALL, "    ")
                 + "\nexcept Exception:\n"
                 + '    raise RuntimeError("H324 FREE PRIVATE DEPENDENCY PREFLIGHT FAILED") from None\n')
    meta = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    dest.mkdir(parents=True, exist_ok=False)
    (dest / SCRIPT).write_text(
        bootstrap + body.replace(future, "", 1), encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "H324_CODE_ONLY_PRIVATE_UNSUBMITTED",
        "kernel": KERNEL_ID, "python_source_files": 1,
        "metadata_files": 1, "media_weights_gold_uploaded": 0,
        "gpu_run_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination), sort_keys=True))

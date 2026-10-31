"""Code-only owner-private H287 free noncommercial Kaggle T4 package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from h270.build_private_payload import INSTALL_H270

KERNEL_TITLE = "Hala H287 Private ShiftDesk LoRA T4"
KERNEL_ID = "abdulazizshehri/hala-h287-private-shiftdesk-lora-t4"
SOURCE_NAME = "private_shiftdesk_lora.py"
SCRIPT_NAME = "h287_private_shiftdesk_lora.py"
INSTALL_H287 = INSTALL_H270.replace(
    "HALA_H270_PRIVATE_NONCOMMERCIAL_FREE_T4",
    "HALA_H287_PRIVATE_NONCOMMERCIAL_FREE_T4",
)


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE_NAME or source.parent.name != "h287"
            or not isinstance(destination, Path) or destination.exists()
            or ".git" in destination.parts
            or destination.is_relative_to(source.parent)):
        raise ValueError("H287 only checked-in source and new external tmp")
    slug = re.sub(r"[^a-z0-9]+", "-", KERNEL_TITLE.lower()).strip("-")
    if KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H287 Kaggle title would create another kernel slug")
    body = source.read_text(encoding="utf-8")
    future = "from __future__ import annotations\n"
    required = (
        "def verified_report(", "def prepare_private_train(",
        "def evaluate_queueboard_pair(", "def verified_free_t4(",
        "def model_prompt(", "if __name__ == \"__main__\":",
        "train_eval_exact_png_disjoint", "official_train_dev_ood_admitted",
    )
    if body.count(future) != 1 or not all(x in body for x in required):
        raise ValueError("H287 private source trust contract changed")
    meta = {
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": SCRIPT_NAME,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT_NAME).write_text(
        future + INSTALL_H287 + "\n" + body.replace(future, "", 1),
        encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "kernel_id": KERNEL_ID, "private": True,
        "free_gpu": "NvidiaTeslaT4",
        "source_scripts": 1, "uploaded_user_media_weights_or_gold": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.destination), sort_keys=True))

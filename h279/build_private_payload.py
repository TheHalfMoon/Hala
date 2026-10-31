"""Create-only code-only private Kaggle H279 payload; no media or model uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from h274.build_private_payload import INSTALL_H274

KERNEL_ID = "abdulazizshehri/hala-h279-private-queueboard-native-t4"
SCRIPT = "h279_private_queueboard_qwen.py"
SOURCE = "private_queueboard_qwen_baseline.py"
INSTALL_H279 = INSTALL_H274.replace(
    "HALA_H274_PRIVATE_FREE_NONCOMMERCIAL",
    "HALA_H279_PRIVATE_FREE_NONCOMMERCIAL")


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE or source.parent.name != "h279"
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H279 checked-in source and fresh scratch only")
    body = source.read_text("utf-8")
    future = "from __future__ import annotations\n"
    if (body.count(future) != 1
            or 'if __name__ == "__main__":' not in body
            or "def validate_rows(" not in body
            or "run_firstparty_queueboard_probe(" not in body
            or "normalize_strict_action(raw)" not in body):
        raise ValueError("H279 private source contract drift")
    metadata = {
        "id": KERNEL_ID, "title": "Hala H279 Private Queueboard Native T4",
        "code_file": SCRIPT, "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT).write_text(
        future + INSTALL_H279 + "\n" + body.replace(future, "", 1),
        encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "private": True, "free_gpu": "NvidiaTeslaT4",
        "uploaded_original_media_or_model": 0, "scripts": 1,
        "kernel_id": KERNEL_ID,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination), sort_keys=True))

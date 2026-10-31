"""H282: build private code-only FREE T4 Kaggle bundle, never model/media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from h273.build_private_payload import INSTALL_H273

KERNEL_ID = "abdulazizshehri/hala-h282-private-pretrained-clinicgrid-t4"\nKERNEL_TITLE = "Hala H282 Private Pretrained ClinicGrid T4"
SOURCE_NAME = "private_clinicgrid_pretrained_probe.py"
SCRIPT_NAME = "h282_private_clinicgrid_native.py"
INSTALL_H282 = INSTALL_H273.replace(
    "HALA_H273_FREE_PRIVATE_NONCOMMERCIAL",
    "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL",
)


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE_NAME or source.parent.name != "h282"
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H282 checked-in source and new temporary destination only")
    text = source.read_text("utf-8")
    future = "from __future__ import annotations\n"
    if (text.count(future) != 1
            or 'if __name__ == "__main__":' not in text
            or "def validate_rows(" not in text
            or "run_original_clinicgrid_probe(" not in text
            or "RelativePrevalidationProcessor(" not in text
            or "screenshot_integrity_attested" not in text):
        raise ValueError("H282 source/payload expected trust contract changed")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": SCRIPT_NAME, "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT_NAME).write_text(
        future + INSTALL_H282 + "\n" + text.replace(future, "", 1),
        encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return {
        "kernel_id": KERNEL_ID, "private": True,
        "free_gpu": "NvidiaTeslaT4", "python_scripts": 1,
        "uploaded_private_media_or_weights": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(build(args.source, args.destination), sort_keys=True))

"""H336: produce ONE private Kaggle T4 code-only script, no dataset/media.

Only builds a local temporary bundle. Does not push or start Kaggle.
An owner must independently check quota, auth, privacy, no concurrent GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

from h282.build_private_payload import INSTALL_H282

TITLE = "Hala H336 First Click Qwen T4"
KERNEL_ID = "abdulazizshehri/hala-h336-first-click-qwen-t4"
SOURCE_NAME = "private_first_click_qwen.py"
SCRIPT_NAME = "h336_private_first_click_qwen.py"
INSTALL = INSTALL_H282.replace(
    "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL",
    "HALA_H336_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4",
)
BOOTSTRAP = """import json as _h336_json
import os as _h336_os
import sys as _h336_sys
from pathlib import Path as _h336_Path
try:
"""
BOOTSTRAP_END = """
except Exception as _err:
    if (_h336_sys.platform == "linux"
            and _h336_Path("/kaggle/working").is_dir()
            and _h336_os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and _h336_os.environ.get(
                "HALA_H336_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4") == "1"):
        _dest = _h336_Path("/kaggle/working") / (
            "hala-h336-private-first-click-stage.json")
        if not _dest.exists():
            _allowed = {"ValueError", "RuntimeError", "OSError", "ImportError",
                        "ModuleNotFoundError", "CalledProcessError",
                        "FileNotFoundError", "OutOfMemoryError", "TimeoutError"}
            _name = type(_err).__name__
            _dest.write_text(_h336_json.dumps({
                "status": "H336_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE",
                "failure_stage": "DEPENDENCIES",
                "exception_class": _name if _name in _allowed
                                   else "OTHER_ERROR_CLASS",
                "source_commit": "ff403d13249ac0f16d806ccfe4b81c4d92df6a07",
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                "model_skill_verified": False, "blind_ood_proven": False,
                "adapted_gain_proven": False,
                "private_media_or_raw_exception_saved": False
            }, sort_keys=True, separators=(",", ":")) + "\\n",
            encoding="utf-8")
    raise RuntimeError("H336 owner-private dependency bootstrap refused") from None
"""


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE_NAME or source.parent.name != "h336"
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H336 only fresh owner-private code bundle")
    slug = re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-")
    if len(TITLE) > 50 or KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H336 private Kaggle kernel slug mismatch")
    raw = source.read_text(encoding="utf-8")
    header = "from __future__ import annotations\n"
    required = (
        'SOURCE_SHA = "ff403d13249ac0f16d806ccfe4b81c4d92df6a07"',
        'SOURCE_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"',
        'KERNEL_FLAG = "HALA_H336_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4"',
        "def clone_pinned_source(", "def validate_aggregate(",
        "def safe_report(", "def safe_failure_receipt(", "def run() -> None:",
        'if __name__ == "__main__":',
    )
    if raw.count(header) != 1 or not all(s in raw for s in required):
        raise ValueError("H336 private data and frozen source integrity contract")
    assembled = (header + BOOTSTRAP + textwrap.indent(INSTALL, "    ")
                 + BOOTSTRAP_END + raw.replace(header, "", 1))
    compile(assembled, SCRIPT_NAME, "exec")
    meta = {
        "id": KERNEL_ID, "title": TITLE, "code_file": SCRIPT_NAME,
        "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT_NAME).write_text(assembled, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "H336_CODE_ONLY_PRIVATE_NONCOMMERCIAL_FREE_T4",
        "private_kernel": KERNEL_ID, "python_source_files": 1,
        "metadata_files": 1, "media_weights_datasets_gold_uploaded": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    options = p.parse_args()
    print(json.dumps(build(options.source, options.destination),
                     sort_keys=True))

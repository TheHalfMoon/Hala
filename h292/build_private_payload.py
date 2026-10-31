"""H292: code-only private Kaggle T4 diagnostic V2 of H287 ERROR V1.

Single Python code file, no data/media/model attached. Fail-closed stage
report on Python dependency bootstrap or runtime failures. Never emit raw
Kaggle traceback text, model output, private images or adapter weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

from h270.build_private_payload import INSTALL_H270

KERNEL_TITLE = "Hala H292 Private ShiftDesk RouteBoard Paired T4"
KERNEL_ID = "abdulazizshehri/hala-h292-private-shiftdesk-routeboard-paired-t4"
SOURCE_NAME = "private_paired_routeboard_dev.py"
SCRIPT_NAME = "h292_private_paired_routeboard_dev.py"
FAIL_REPORT = "hala-h292-private-routeboard-stage-diagnostic.json"
INSTALL_H292 = INSTALL_H270.replace(
    "HALA_H270_PRIVATE_NONCOMMERCIAL_FREE_T4",
    "HALA_H292_PRIVATE_NONCOMMERCIAL_FREE_T4",
)

# Wrap original pinned Kaggle Python/CUDA dependency installer without
# ever displaying its exception or writing a raw log. This catches failures
# even BEFORE the actual Qwen driver can import.
BOOTSTRAP_PREFIX = """import json as _diag_json
import os as _diag_os
import sys as _diag_sys
from pathlib import Path as _diag_Path
try:
"""
BOOTSTRAP_SUFFIX = """
except Exception as _diag_exc:
    if (_diag_sys.platform == 'linux'
            and _diag_Path('/kaggle/working').is_dir()
            and _diag_os.environ.get('KAGGLE_KERNEL_RUN_TYPE')
            and _diag_os.environ.get(
                'HALA_H292_PRIVATE_NONCOMMERCIAL_FREE_T4') == '1'):
        _diag_allowed = {'RuntimeError', 'ValueError', 'ImportError',
                         'ModuleNotFoundError', 'OSError',
                         'CalledProcessError', 'FileNotFoundError'}
        _diag_cls = type(_diag_exc).__name__
        _diag_data = {
            'status': 'H292_PRIVATE_FAILURE_DIAGNOSTIC_NOT_A_MODEL_SCORE',
            'source_protocol': 'H287_ORIGINAL_SHIFT_TRAIN_QUEUE_RESEARCH',
            'previous_v1_terminal_error': True,
            'failure_stage': 'BOOTSTRAP',
            'exception_class': (_diag_cls if _diag_cls in _diag_allowed
                                else 'OTHER_ERROR_CLASS'),
            'model_id': 'Qwen/Qwen3-VL-2B-Instruct',
            'model_revision': '89644892e4d85e24eaac8bacfd4f463576704203',
            'optimizer_steps_verified': 0,
            'paired_gui_task_score_verified': False,
            'official_training_admitted': False,
            'independent_blind_ood_verified': False,
            'model_improvement_proven': False,
            'raw_exception_logs_images_points_or_weights_saved': False,
        }
        _diag_dest = _diag_Path('/kaggle/working') / (
            'hala-h292-private-routeboard-stage-diagnostic.json')
        if not _diag_dest.exists():
            _diag_dest.write_text(
                _diag_json.dumps(_diag_data, sort_keys=True) + '\\n',
                encoding='utf-8')
    raise RuntimeError('H292 private bootstrap refused; read only '
                       'bounded stage diagnostic') from None
"""


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or source.name != SOURCE_NAME
            or source.parent.name != "h292" or not source.is_file()
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H292 exact checked-in source and fresh tmp only")
    slug = re.sub(r"[^a-z0-9]+", "-", KERNEL_TITLE.lower()).strip("-")
    if KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H292 owner-private Kaggle title/ID mismatch")
    body = source.read_text("utf-8")
    future = "from __future__ import annotations\n"
    markers = (
        'DIAGNOSTIC_REPORT = "hala-h292-private-routeboard-stage-diagnostic.json"',
        'STAGE_ENUM = frozenset(', "def failure_receipt(",
        "def write_private_failure_receipt(", "def verified_report(",
        "def evaluate_queueboard_pair(", "def evaluate_routeboard_dev(", "if __name__ == \"__main__\":",
    )
    if body.count(future) != 1 or not all(x in body for x in markers):
        raise ValueError("H292 stage and native research contract drift")
    bundle = (future + BOOTSTRAP_PREFIX
              + textwrap.indent(INSTALL_H292, "    ")
              + BOOTSTRAP_SUFFIX
              + body.replace(future, "", 1))
    metadata = {
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": SCRIPT_NAME,
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / SCRIPT_NAME).write_text(bundle, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {
        "private": True, "free_t4": True,
        "kernel_id": KERNEL_ID,
        "upload_files": 2, "uploaded_media_weights_or_gold": 0,
        "bootstrap_diagnostic_bounded": True,
        "runtime_diagnostic_bounded": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.destination), sort_keys=True))

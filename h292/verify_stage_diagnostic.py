"""H292 parse ONLY owner-private bounded failure stage, never raw Kaggle log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h280.verify_private_native_aggregate import (
    EvidenceRejected, decode_bounded, exact_keys, need, flag, count,
)
from h292.private_paired_routeboard_dev import (
    DIAGNOSTIC_REPORT, STAGE_ENUM, MODEL, MODEL_REV,
)

FIELDS = frozenset({
    "status", "source_protocol", "previous_v1_terminal_error",
    "failure_stage", "exception_class", "model_id", "model_revision",
    "optimizer_steps_verified", "paired_gui_task_score_verified",
    "official_training_admitted", "independent_blind_ood_verified",
    "model_improvement_proven",
    "raw_exception_logs_images_points_or_weights_saved",
})
ALLOWED_CLASSES = frozenset({
    "RuntimeError", "ValueError", "TypeError", "ImportError",
    "ModuleNotFoundError", "FileNotFoundError", "OSError",
    "AssertionError", "KeyError", "IndexError", "AttributeError",
    "MemoryError", "OutOfMemoryError", "CalledProcessError",
    "OTHER_ERROR_CLASS",
})


def check_stage_receipt(payload: object) -> dict:
    data = exact_keys(payload, FIELDS, "H292 extra/private diagnostic field")
    need(type(data["status"]) is str and data["status"] ==
         "H292_PRIVATE_FAILURE_DIAGNOSTIC_NOT_A_MODEL_SCORE",
         "H292 expected failed experiment, no score")
    need(type(data["source_protocol"]) is str
         and data["source_protocol"] ==
         "H287_ORIGINAL_SHIFT_TRAIN_QUEUE_RESEARCH",
         "H292 original protocol mismatch")
    need(type(data["model_id"]) is str and data["model_id"] == MODEL
         and type(data["model_revision"]) is str
         and data["model_revision"] == MODEL_REV,
         "H292 original Qwen model identity mismatch")
    stage = data["failure_stage"]
    need(type(stage) is str and stage in STAGE_ENUM
         and stage != "SUCCESS",
         "H292 unsupported/success-stage diagnostic")
    klass = data["exception_class"]
    need(type(klass) is str and klass in ALLOWED_CLASSES,
         "H292 unsupported exception category")
    flag(data["previous_v1_terminal_error"], True,
         "H292 V1 error provenance missing")
    count(data["optimizer_steps_verified"], 0,
          "H292 failure report MUST NOT claim verified optimizer updates")
    for key in (
        "paired_gui_task_score_verified", "official_training_admitted",
        "independent_blind_ood_verified", "model_improvement_proven",
        "raw_exception_logs_images_points_or_weights_saved",
    ):
        flag(data[key], False, "H292 invalid failure performance/privacy claim")
    return {
        "status": "H292_SAFE_FAILURE_STAGE_ONLY_NO_MODEL_SCORE",
        "failure_stage": stage,
        "exception_class": klass,
        "verified_optimizer_steps": 0,
        "verified_paired_gui_scores": False,
        "official_training_admitted": False,
        "blind_ood_proven": False,
        "generalizable_gain_proven": False,
    }


def verify_bytes(data: bytes) -> dict:
    return check_stage_receipt(decode_bounded(data))


def verify_private_file(path: Path) -> dict:
    need(isinstance(path, Path) and path.name == DIAGNOSTIC_REPORT
         and path.is_file() and not path.is_symlink(),
         "H292 only exact whitelisted private diagnostic basename")
    try:
        n = path.stat().st_size
        need(100 <= n <= 1200, "H292 oversized diagnostic prohibited")
        return verify_bytes(path.read_bytes())
    except OSError:
        raise EvidenceRejected("H292 private diagnostic inaccessible") from None


def main() -> None:
    p = argparse.ArgumentParser(description="H292 bounded stage only, never logs")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    try:
        output = verify_private_file(args.report)
    except EvidenceRejected as exc:
        p.exit(2, str(exc) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()

"""H333 strict private bounded aggregate verifier; never prints raw receipt.

Receipt alone cannot independently attest a genuine owner Kaggle job,
the exact submitted source code, pretrained model weights or GPU.
"""
from __future__ import annotations

from pathlib import Path
import json
from h333.private_paired_exposed_qwen import (
    SOURCE_SHA, SOURCE_BLOB, MODEL_ID, MODEL_REV, REPORT, DIAG, STATUS,
    DIAG_STATUS, STAGES, EXCEPTIONS, validate_aggregate,
)


def need(value: bool) -> None:
    if not value:
        raise ValueError("H333 STRICT_PRIVATE_RECEIPT_REFUSED")


def unique_pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        need(key not in result)
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError("H333 STRICT_PRIVATE_RECEIPT_REFUSED")


def decode(raw: object) -> dict:
    need(type(raw) is bytes and 20 <= len(raw) <= 4096)
    try:
        result = json.loads(raw.decode("utf-8"),
                            object_pairs_hook=unique_pairs,
                            parse_constant=reject_constant)
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("H333 STRICT_PRIVATE_RECEIPT_REFUSED") from None
    need(type(result) is dict)
    return result


def verify_success(data: object) -> dict:
    need(type(data) is dict and set(data) == {
        "status", "source_commit", "source_html_git_blob",
        "model_id", "model_revision", "hardware",
        "exposed_paired_source_aggregate", "new_optimizer_steps",
        "official_train_dev_ood_admitted", "independent_blind_ood",
        "trained_model_gain_proven",
        "private_media_model_text_gold_coordinates_or_weights_saved",
    })
    for key, expected in (
        ("status", STATUS), ("source_commit", SOURCE_SHA),
        ("source_html_git_blob", SOURCE_BLOB), ("model_id", MODEL_ID),
        ("model_revision", MODEL_REV), ("new_optimizer_steps", 0),
        ("official_train_dev_ood_admitted", False),
        ("independent_blind_ood", False),
        ("trained_model_gain_proven", False),
        ("private_media_model_text_gold_coordinates_or_weights_saved", False),
    ):
        need(type(data[key]) is type(expected) and data[key] == expected)
    hardware = data["hardware"]
    need(type(hardware) is dict and set(hardware) == {
        "free_owner_private_t4", "t4_count"}
         and hardware["free_owner_private_t4"] is True
         and type(hardware["t4_count"]) is int
         and hardware["t4_count"] in (1, 2))
    validate_aggregate(data["exposed_paired_source_aggregate"])
    aggregate = data["exposed_paired_source_aggregate"]["by_language"]
    return {
        "status": "H333_STRICT_EXPOSED_PRETRAINED_PROMPT_STUDY_ACCEPTED",
        "public_exposed_pairs": 4, "native_model_runs_claimed": 8,
        "en_control_successes": aggregate["en"]["control"]["task_success"],
        "en_candidate_successes": aggregate["en"]["candidate"]["task_success"],
        "ar_control_successes": aggregate["ar"]["control"]["task_success"],
        "ar_candidate_successes": aggregate["ar"]["candidate"]["task_success"],
        "independent_owner_private_job_attestation_required": True,
        "fresh_blind_ood": False, "new_optimizer_steps": 0,
        "adapted_gain_proven": False, "can_release": False,
    }


def verify_failure(data: object) -> dict:
    need(type(data) is dict and set(data) == {
        "status", "failure_stage", "exception_class",
        "source_commit", "model_id", "model_revision",
        "model_skill_verified", "blind_ood_proven",
        "adapted_gain_proven", "private_media_or_raw_exception_saved",
    })
    for key, expected in (
        ("status", DIAG_STATUS), ("source_commit", SOURCE_SHA),
        ("model_id", MODEL_ID), ("model_revision", MODEL_REV),
        ("model_skill_verified", False), ("blind_ood_proven", False),
        ("adapted_gain_proven", False),
        ("private_media_or_raw_exception_saved", False),
    ):
        need(type(data[key]) is type(expected) and data[key] == expected)
    need(type(data["failure_stage"]) is str
         and data["failure_stage"] in STAGES
         and type(data["exception_class"]) is str
         and data["exception_class"] in (EXCEPTIONS | {"OTHER_ERROR_CLASS"}))
    return {"status": "H333_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE",
            "failure_stage": data["failure_stage"],
            "model_skill_verified": False, "can_release": False}


def verify_file(path: Path, *, diagnostic: bool = False) -> dict:
    expected = DIAG if diagnostic else REPORT
    need(isinstance(path, Path) and path.name == expected
         and path.is_file() and not path.is_symlink())
    try:
        return (verify_failure if diagnostic else verify_success)(
            decode(path.read_bytes()))
    except OSError:
        raise ValueError("H333 STRICT_PRIVATE_RECEIPT_REFUSED") from None

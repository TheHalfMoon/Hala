"""H336 strict allowlist receipt parser for owner-private first-click diagnosis.

Read only the exact whitelisted small JSON report after external job/status,
source, model and privacy attestation. Never print the source JSON, hidden
gold, model points, images or other Kaggle output files.
"""
from __future__ import annotations

import json
from pathlib import Path

from h336.private_first_click_qwen import (
    DIAG, DIAG_STATUS, EXCEPTIONS, MODEL_ID, MODEL_REV,
    REPORT, SOURCE_BLOB, SOURCE_SHA, STAGES, STATUS,
    validate_aggregate,
)


def need(value: bool) -> None:
    if not value:
        raise ValueError("H336 STRICT_OWNER_PRIVATE_RECEIPT_REFUSED")


def _unique_pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, val in items:
        need(key not in result)
        result[key] = val
    return result


def _reject_constant(_: object):
    raise ValueError("H336 STRICT_OWNER_PRIVATE_RECEIPT_REFUSED")


def parse_json(raw: object) -> dict:
    need(type(raw) is bytes and 50 <= len(raw) <= 4096)
    try:
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("H336 STRICT_OWNER_PRIVATE_RECEIPT_REFUSED") from None
    need(type(data) is dict)
    return data


def verify_success(data: object) -> dict:
    need(type(data) is dict and set(data) == {
        "status", "source_commit", "source_html_git_blob",
        "model_id", "model_revision", "hardware",
        "source_exposed_first_click_counts", "new_optimizer_steps",
        "official_train_admitted", "fresh_blind_ood",
        "trained_model_gain_proven", "can_release",
        "raw_private_model_actions_media_gold_coordinates_saved",
    })
    for key, expected in (
        ("status", STATUS), ("source_commit", SOURCE_SHA),
        ("source_html_git_blob", SOURCE_BLOB),
        ("model_id", MODEL_ID), ("model_revision", MODEL_REV),
        ("new_optimizer_steps", 0), ("official_train_admitted", False),
        ("fresh_blind_ood", False), ("trained_model_gain_proven", False),
        ("can_release", False),
        ("raw_private_model_actions_media_gold_coordinates_saved", False),
    ):
        need(type(data[key]) is type(expected) and data[key] == expected)
    hw = data["hardware"]
    need(type(hw) is dict and set(hw) == {
        "free_owner_private_t4", "t4_count"}
        and hw["free_owner_private_t4"] is True
        and type(hw["t4_count"]) is int and hw["t4_count"] in (1, 2))
    verified = validate_aggregate(data["source_exposed_first_click_counts"])
    return {
        "status": "H336_STRICT_EXPOSED_FIRST_CLICK_CATEGORY_REPORT_ACCEPTED",
        "public_exposed_scenes": 12,
        "model_generations_claimed": verified["first_click_model_generations"],
        "by_language": verified["by_language"],
        "owner_private_job_attestation_required": True,
        "trained_model_gain_proven": False,
        "fresh_blind_ood": False, "can_release": False,
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
    return {
        "status": DIAG_STATUS, "failure_stage": data["failure_stage"],
        "model_skill_verified": False, "can_release": False,
    }


def verify_file(path: Path, *, diagnostic: bool = False) -> dict:
    target = DIAG if diagnostic else REPORT
    need(isinstance(path, Path) and path.name == target
         and path.is_file() and not path.is_symlink())
    try:
        raw = path.read_bytes()
    except OSError:
        raise ValueError("H336 STRICT_OWNER_PRIVATE_RECEIPT_REFUSED") from None
    return (verify_failure if diagnostic else verify_success)(parse_json(raw))

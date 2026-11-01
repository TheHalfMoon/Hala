"""H324 independent strict verifier for owner-PRIVATE low-entropy receipt.

Never prints raw browser actions, screenshots, points, prompts or model text.
Owner Kaggle kernel identity and exact submitted code must be checked
separately; a JSON file alone does not authenticate Qwen provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h324.private_pretrained_beaconboard_two_case import (
    SOURCE_SHA, SOURCE_BLOB, MODEL_ID, MODEL_REV,
    STATUS, DIAG_STATUS, REPORT, DIAG, STAGES, EXCEPTIONS,
    strict_row,
)


class EvidenceRejected(ValueError):
    """Fixed message only: do not echo private contents."""


def need(ok: bool) -> None:
    if not ok:
        raise EvidenceRejected("H324 PRIVATE_AGGREGATE_REJECTED")


def unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        need(key not in obj)
        obj[key] = value
    return obj


def reject_constant(_value):
    raise EvidenceRejected("H324 PRIVATE_AGGREGATE_REJECTED")


def decode(raw: bytes) -> object:
    need(type(raw) is bytes and 10 < len(raw) <= 4096)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant)
    except (UnicodeError, ValueError, TypeError):
        raise EvidenceRejected("H324 PRIVATE_AGGREGATE_REJECTED") from None


TOP = frozenset({
    "status", "source_commit", "source_html_git_blob", "model_id",
    "model_revision", "hardware", "research_cases",
    "model_generations", "native_actions_dispatched", "task_successes",
    "rows", "public_exposed_not_blind", "pretrained_optimizer_steps",
    "official_train_dev_ood_admitted", "independent_blind_ood",
    "adapted_model_gain_proven",
    "raw_media_gold_model_output_coordinates_or_weights_saved",
})
DIAG_KEYS = frozenset({
    "status", "failure_stage", "exception_class", "source_commit",
    "model_id", "model_revision", "model_skill_verified",
    "official_train_dev_ood_admitted", "blind_ood_proven",
    "private_media_or_raw_exception_saved",
})


def verify_success(data: object) -> dict:
    need(type(data) is dict and set(data) == TOP)
    for key, expected in (
        ("status", STATUS), ("source_commit", SOURCE_SHA),
        ("source_html_git_blob", SOURCE_BLOB), ("model_id", MODEL_ID),
        ("model_revision", MODEL_REV), ("public_exposed_not_blind", True),
        ("pretrained_optimizer_steps", 0),
        ("official_train_dev_ood_admitted", False),
        ("independent_blind_ood", False),
        ("adapted_model_gain_proven", False),
        ("raw_media_gold_model_output_coordinates_or_weights_saved", False),
        ("research_cases", 2),
    ):
        need(type(data[key]) is type(expected) and data[key] == expected)
    hardware = data["hardware"]
    need(type(hardware) is dict and set(hardware) == {"free_t4", "devices"}
         and hardware["free_t4"] is True
         and type(hardware["devices"]) is int
         and hardware["devices"] in (1, 2))
    rows = data["rows"]
    need(type(rows) is list and len(rows) == 2)
    try:
        clean = [strict_row(row, lang) for row, lang in
                 zip(rows, ("en", "ar"), strict=True)]
    except Exception:
        raise EvidenceRejected("H324 PRIVATE_AGGREGATE_REJECTED") from None
    counts = (
        ("model_generations", sum(row["model_generations"] for row in clean)),
        ("native_actions_dispatched",
         sum(row["native_dispatched"] for row in clean)),
        ("task_successes", sum(int(row["task_success"]) for row in clean)),
    )
    for key, actual in counts:
        need(type(data[key]) is int and data[key] == actual)
    return {
        "status": "H324_STRICT_PRIVATE_EXPOSED_PRETRAINED_RECEIPT_ACCEPTED",
        "public_original_research_cases": 2,
        "genuine_model_steps_claimed": data["model_generations"],
        "native_actions_dispatched": data["native_actions_dispatched"],
        "source_task_successes": data["task_successes"],
        "owner_private_kernel_identity_external_verification_required": True,
        "public_source_exposed_not_blind": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "adapted_gain_proven": False,
    }


def verify_failure(data: object) -> dict:
    need(type(data) is dict and set(data) == DIAG_KEYS)
    for key, expected in (
        ("status", DIAG_STATUS), ("source_commit", SOURCE_SHA),
        ("model_id", MODEL_ID), ("model_revision", MODEL_REV),
        ("model_skill_verified", False),
        ("official_train_dev_ood_admitted", False),
        ("blind_ood_proven", False),
        ("private_media_or_raw_exception_saved", False),
    ):
        need(type(data[key]) is type(expected) and data[key] == expected)
    need(type(data["failure_stage"]) is str
         and data["failure_stage"] in STAGES
         and type(data["exception_class"]) is str
         and data["exception_class"] in (EXCEPTIONS | {"OTHER_ERROR_CLASS"}))
    return {
        "status": "H324_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE",
        "stage": data["failure_stage"], "class": data["exception_class"],
        "owner_private_kernel_identity_external_verification_required": True,
        "genuine_model_skill_verified": False,
        "blind_ood_proven": False,
    }


def verify_file(path: Path, *, diagnostic: bool = False) -> dict:
    expected = DIAG if diagnostic else REPORT
    need(isinstance(path, Path) and path.name == expected
         and path.is_file() and not path.is_symlink())
    try:
        payload = decode(path.read_bytes())
        return (verify_failure if diagnostic else verify_success)(payload)
    except OSError:
        raise EvidenceRejected("H324 PRIVATE_AGGREGATE_REJECTED") from None


def main() -> None:
    p = argparse.ArgumentParser(description="H324 private aggregate verifier")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    try:
        safe = verify_file(args.report, diagnostic=args.diagnostic)
    except EvidenceRejected:
        p.exit(2, "H324 PRIVATE_AGGREGATE_REJECTED\n")
    print(json.dumps(safe, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()

"""H354 bounded no-sensitive-data result: native paired arithmetic reused."""
from __future__ import annotations
from pathlib import Path

from h353.verify_private_receipt import (
    parse_unique, verify_success as prior_success,
    verify_stage_only as prior_stage,
)
from h353.build_private_payload import STATUS as PREV_STATUS
from h354.build_private_payload import STATUS, REPORT, DIAG


def need(value):
    if not value:
        raise ValueError("H354 PRIVATE RECEIPT REFUSED")


def verify_success(data):
    need(type(data) is dict and data.get("status") == STATUS)
    old = dict(data)
    old["status"] = PREV_STATUS
    evidence = prior_success(old)
    return {
        "status": "H354_STRICT_ORDER_BALANCED_CONSERVATIVE_RECEIPT_ACCEPTED",
        "by_language": evidence["by_language"],
        "claimed_model_generations": evidence["claimed_model_generations"],
        "owner_private_kaggle_code_attestation_required": True,
        "new_trained_model": False,
        "source_exposed_not_blind": True,
        "can_release": False,
    }


def verify_stage_only(data):
    need(type(data) is dict and data.get("status") ==
         "H354_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE")
    old = dict(data)
    old["status"] = "H353_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE"
    accepted = prior_stage(old)
    return {"status": data["status"], "stage": accepted["stage"],
            "can_release": False}


def verify_file(path: Path, *, diagnostic: bool = False):
    need(isinstance(path, Path) and path.is_file() and not path.is_symlink()
         and path.name == (DIAG if diagnostic else REPORT))
    data = parse_unique(path.read_bytes())
    return (verify_stage_only if diagnostic else verify_success)(data)

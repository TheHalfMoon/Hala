"""H353 strict private paired receipt, reusing tested H352 arithmetic gates.

Never publish original JSON, per-case model text, click coordinates or HTML.
Kaggle privacy, job version and code hash require independent attestation.
"""
from __future__ import annotations
from pathlib import Path
from h352.verify_private_receipt import (
    parse_unique, verify_success as verify_h352_success,
    verify_stage_only as verify_h352_stage,
)
from h352.private_qwen_paired_snap import STATUS as PREVIOUS_STATUS
from h353.build_private_payload import STATUS, REPORT, DIAG

def need(ok):
    if not ok:
        raise ValueError("H353 STRICT PRIVATE RECEIPT REFUSED")

def verify_success(data):
    need(type(data) is dict and data.get("status") == STATUS)
    adjusted = dict(data)
    adjusted["status"] = PREVIOUS_STATUS
    prior = verify_h352_success(adjusted)
    return {
        "status": "H353_STRICT_COUNTERBALANCED_PAIRED_RECEIPT_ACCEPTED",
        "by_language": prior["by_language"],
        "claimed_model_generations": prior["claimed_model_generations"],
        "requires_external_owner_kaggle_code_attestation": True,
        "public_exposed_source_not_blind": True,
        "adapted_model_gain_proven": False, "can_release": False,
    }

def verify_stage_only(data):
    need(type(data) is dict
         and data.get("status") == "H353_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE")
    changed = dict(data)
    changed["status"] = "H352_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE"
    result = verify_h352_stage(changed)
    return {"status": data["status"], "stage": result["stage"],
            "can_release": False}

def verify_file(path: Path, *, diagnostic=False):
    expected = DIAG if diagnostic else REPORT
    need(isinstance(path, Path) and path.name == expected
         and path.is_file() and not path.is_symlink())
    data = parse_unique(path.read_bytes())
    return (verify_stage_only if diagnostic else verify_success)(data)

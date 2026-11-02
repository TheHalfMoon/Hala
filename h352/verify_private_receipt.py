"""H352 only-authorized private paired category receipt verifier.

Must additionally attest owner-private exact Kaggle job/code hash externally.
Never download or disclose model text, point, media, per-case data, logs.
"""
from __future__ import annotations
import json
from pathlib import Path

from h352.private_qwen_paired_snap import (
    STATUS, PIN, BLOB, MODEL, REV, REPORT, DIAG, STAGES,
)


def need(ok):
    if not ok:
        raise ValueError("H352 STRICT PRIVATE RECEIPT REFUSED")


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result)
        result[key] = value
    return result


def _constant(_):
    raise ValueError("H352 STRICT PRIVATE RECEIPT REFUSED")


def parse_unique(raw):
    need(type(raw) is bytes and 50 <= len(raw) <= 4096)
    try:
        result = json.loads(raw.decode("utf-8"),
                            object_pairs_hook=_pairs,
                            parse_constant=_constant)
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("H352 STRICT PRIVATE RECEIPT REFUSED") from None
    need(type(result) is dict)
    return result


def verify_success(data):
    need(type(data) is dict and set(data) == {
        "status", "source_commit", "source_html_blob",
        "model_id", "model_revision", "hardware", "by_language",
        "claimed_wrapped_model_generations",
        "paired_source_exposed_not_blind",
        "private_job_attestation_required", "official_train_admitted",
        "fresh_blind_ood", "adapted_model_gain_proven",
        "new_optimizer_steps", "can_release",
        "raw_predictions_or_media_exported",
    })
    expected = {
        "status": STATUS, "source_commit": PIN,
        "source_html_blob": BLOB, "model_id": MODEL,
        "model_revision": REV,
        "claimed_wrapped_model_generations": 12,
        "paired_source_exposed_not_blind": True,
        "private_job_attestation_required": True,
        "official_train_admitted": False,
        "fresh_blind_ood": False, "adapted_model_gain_proven": False,
        "new_optimizer_steps": 0, "can_release": False,
        "raw_predictions_or_media_exported": False,
    }
    for key, wanted in expected.items():
        need(type(data[key]) is type(wanted) and data[key] == wanted)
    hw = data["hardware"]
    need(type(hw) is dict
         and set(hw) == {"free_owner_private_t4", "t4_count"}
         and hw["free_owner_private_t4"] is True
         and type(hw["t4_count"]) is int
         and hw["t4_count"] in (1, 2))
    groups = data["by_language"]
    need(type(groups) is dict and set(groups) == {"en", "ar"})
    keys = {
        "scenes", "baseline_success", "pixel_snap_success",
        "pixel_snap_changes", "paired_wins", "paired_losses",
        "ties_both_success", "ties_both_fail",
    }
    for language in ("en", "ar"):
        group = groups[language]
        need(type(group) is dict and set(group) == keys
             and all(type(n) is int and 0 <= n <= 6
                     for n in group.values())
             and group["scenes"] == 6
             and group["paired_wins"] + group["paired_losses"]
                 + group["ties_both_success"] + group["ties_both_fail"] == 6
             and group["baseline_success"]
                 == group["paired_losses"] + group["ties_both_success"]
             and group["pixel_snap_success"]
                 == group["paired_wins"] + group["ties_both_success"]
             and group["pixel_snap_success"]
                 == group["baseline_success"] + group["paired_wins"]
                    - group["paired_losses"])
    return {
        "status": "H352_STRICT_EXPOSED_REAL_QWEN_PAIRED_RECEIPT_ACCEPTED",
        "by_language": groups,
        "claimed_model_generations": 12,
        "external_kaggle_job_attestation_required": True,
        "public_source_exposed_not_blind": True,
        "adapted_model_gain_proven": False, "can_release": False,
    }


def verify_stage_only(data):
    need(type(data) is dict and set(data) == {
        "status", "stage", "can_release", "fresh_blind_ood",
        "adapted_model_gain_proven", "official_train_admitted",
        "raw_private_data_saved",
    })
    need(data["status"] == "H352_PRIVATE_STAGE_ONLY_NOT_MODEL_SCORE"
         and type(data["stage"]) is str and data["stage"] in STAGES
         and all(data[key] is False for key in (
             "can_release", "fresh_blind_ood",
             "adapted_model_gain_proven", "official_train_admitted",
             "raw_private_data_saved")))
    return {"status": data["status"], "stage": data["stage"],
            "can_release": False}


def verify_file(path: Path, *, diagnostic: bool = False):
    name = DIAG if diagnostic else REPORT
    need(isinstance(path, Path) and path.name == name
         and path.is_file() and not path.is_symlink())
    return (verify_stage_only if diagnostic else verify_success)(
        parse_unique(path.read_bytes()))

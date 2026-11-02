"""Strict bounded H351 PRIVATE receipt reader (one allowlisted aggregate only).

A well-formed JSON report is NOT independent proof of the Kaggle/model job.
Do not access output directories or log raw predictions, images or points.
"""
from __future__ import annotations
import json
from pathlib import Path

from h337.source_miss_distance_bands import CATEGORIES
from h351.private_qwen_miss_distance import (
    STATUS, PIN, HTML_BLOB, MODEL_ID, MODEL_REV, REPORT, ERROR_FILE,
)

def refuse(ok):
    if not ok:
        raise ValueError("H351 PRIVATE RECEIPT REFUSED")

def parse_unique(raw):
    refuse(type(raw) is bytes and 30 <= len(raw) <= 4096)
    def pairs(items):
        result = {}
        for key, value in items:
            refuse(key not in result)
            result[key] = value
        return result
    def constant(_):
        raise ValueError("H351 PRIVATE RECEIPT REFUSED")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=constant)
    except (UnicodeError, TypeError, ValueError):
        raise ValueError("H351 PRIVATE RECEIPT REFUSED") from None
    refuse(type(value) is dict)
    return value

def verify_success(value):
    refuse(type(value) is dict and set(value) == {
        "status", "source_commit", "source_html_blob",
        "model_id", "model_revision", "hardware", "by_language",
        "claimed_wrapped_model_generations",
        "original_source_public_exposed",
        "model_identity_requires_separate_job_attestation",
        "official_train_admitted", "fresh_blind_ood",
        "adapted_model_gain_proven", "new_optimizer_steps",
        "can_release", "raw_predictions_or_media_exported",
    })
    expected = {
        "status": STATUS, "source_commit": PIN,
        "source_html_blob": HTML_BLOB, "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "claimed_wrapped_model_generations": 12,
        "original_source_public_exposed": True,
        "model_identity_requires_separate_job_attestation": True,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "new_optimizer_steps": 0,
        "can_release": False, "raw_predictions_or_media_exported": False,
    }
    for key, wanted in expected.items():
        refuse(type(value[key]) is type(wanted) and value[key] == wanted)
    hardware = value["hardware"]
    refuse(type(hardware) is dict
           and set(hardware) == {"free_owner_private_t4", "t4_count"}
           and hardware["free_owner_private_t4"] is True
           and type(hardware["t4_count"]) is int
           and hardware["t4_count"] in (1, 2))
    groups = value["by_language"]
    refuse(type(groups) is dict and set(groups) == {"en", "ar"})
    for language in ("en", "ar"):
        row = groups[language]
        refuse(type(row) is dict
               and set(row) == {"scenes", "native_actions", "categories"}
               and type(row["scenes"]) is int and row["scenes"] == 6)
        cats = row["categories"]
        refuse(type(cats) is dict and set(cats) == set(CATEGORIES)
               and all(type(n) is int and 0 <= n <= 6
                       for n in cats.values())
               and sum(cats.values()) == 6
               and type(row["native_actions"]) is int
               and row["native_actions"] == 6 - cats["abstain"])
    # Only broad-category counts; no raw model text/points/images may appear.
    return {
        "status": "H351_STRICT_EXPOSED_MISS_DISTANCE_RECEIPT_ACCEPTED",
        "by_language": groups, "source_exposed": True,
        "claimed_model_generations": 12,
        "requires_external_kaggle_job_attestation": True,
        "adapted_model_gain_proven": False, "fresh_blind_ood": False,
        "can_release": False,
    }

def verify_stage_only(value):
    refuse(type(value) is dict and set(value) == {
        "status", "stage", "official_train_admitted",
        "adapted_gain_proven", "fresh_blind_ood", "can_release",
        "raw_private_model_data_saved",
    })
    from h351.private_qwen_miss_distance import STAGES
    refuse(value["status"] == "H351_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE"
           and type(value["stage"]) is str and value["stage"] in STAGES
           and all(value[key] is False for key in (
               "official_train_admitted", "adapted_gain_proven",
               "fresh_blind_ood", "can_release",
               "raw_private_model_data_saved")))
    return {"status": value["status"], "stage": value["stage"],
            "can_release": False}

def verify_file(path, *, diagnostic=False):
    name = ERROR_FILE if diagnostic else REPORT
    refuse(isinstance(path, Path) and path.name == name
           and path.is_file() and not path.is_symlink())
    raw = path.read_bytes()
    return (verify_stage_only if diagnostic else verify_success)(
        parse_unique(raw))

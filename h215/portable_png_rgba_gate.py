"""H215: cross-platform raster parity without obscuring PNG codec drift.

Ubuntu CI actually reported 96/96 different PNG byte digests but 0/96
different decoded RGBA pixel digests from the original Windows source.
This RESEARCH gate anchors frozen Windows metadata bytes to an immutable
SHA256 and checks 96 exact pixel digests plus all other lineage fields.
No Hala TRAIN/DEV/OOD admission, owner media, model or trained weights.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h212.procedural_native_gui_candidates import COUNT, SCHEMA

FROZEN_WINDOWS_JSON_SHA256 = (
    "6a008992c51f6fd01d37fa321bf9f5fa1a346d8d8fb506cb52f3abf001111c8b"
)
MAX_BYTES = 64 * 1024


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H215 pinned original GUI source: " + reason)


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate source JSON key")
        result[key] = value
    return result


def reject_constant(value: str):
    raise ValueError("H215 pinned original GUI source: invalid JSON constant")


def verify_linux_frozen_raster(frozen_bytes: bytes, regenerated: dict) -> dict:
    """Allow PNG encoding differences only under independent frozen SHA pin."""
    require(type(frozen_bytes) is bytes and 0 < len(frozen_bytes) <= MAX_BYTES
            and type(regenerated) is dict,
            "bounded immutable source bytes and generated record required")
    require(sha256(frozen_bytes).hexdigest() == FROZEN_WINDOWS_JSON_SHA256,
            "Windows frozen source metadata changed (including PNG hashes)")
    try:
        doc = json.loads(
            frozen_bytes.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (ValueError, UnicodeError):
        raise ValueError("H215 pinned original GUI source: invalid metadata") from None
    require(type(doc) is dict and doc.get("schema") == SCHEMA
            and doc.get("official_admitted") is False
            and doc.get("dev_gold_sealed") is False
            and doc.get("official_ood_evaluated") is False
            and type(doc.get("registered_candidate_count")) is int
            and doc["registered_candidate_count"] == COUNT
            and not ADMITTED_TRAIN_MANIFEST_SHAS,
            "research-only source flags or identity changed")
    frozen_cases, new_cases = doc.get("cases"), regenerated.get("cases")
    require(type(frozen_cases) is list and type(new_cases) is list
            and len(frozen_cases) == len(new_cases) == COUNT,
            "exact 96 ordered research candidates required")
    local_pngs, original_pngs = [], []
    comparable = dict(regenerated)
    comparable["cases"] = []
    for original, new in zip(frozen_cases, new_cases):
        require(type(original) is dict and type(new) is dict
                and set(original) == set(new)
                and "image_png_sha256" in original
                and "image_decoded_rgba_sha256" in original,
                "unexpected source candidate fields")
        for field in ("image_png_sha256", "image_decoded_rgba_sha256"):
            a, b = original[field], new[field]
            require(type(a) is str and type(b) is str
                    and len(a) == len(b) == 64
                    and all(ch in "0123456789abcdef" for ch in a+b),
                    "invalid source digest")
        require(new["image_decoded_rgba_sha256"] ==
                original["image_decoded_rgba_sha256"],
                "actual decoded RGBA pixels changed")
        local_pngs.append(new["image_png_sha256"])
        original_pngs.append(original["image_png_sha256"])
        # Substitution of computed PNG digest is permitted ONLY because
        # the original manifest bytes have an independent immutable SHA
        # check above. We never replace, update or admit old PNG hashes.
        candidate = dict(new)
        candidate["image_png_sha256"] = original["image_png_sha256"]
        comparable["cases"].append(candidate)
    require(len(set(local_pngs)) == len(set(original_pngs)) == COUNT,
            "generated PNG identity collision")
    require(comparable == doc,
            "generator-source, role, language or release metadata changed")
    mismatch = sum(a != b for a, b in zip(local_pngs, original_pngs))
    return {
        "status": "H215_PINNED_WINDOWS_METADATA_AND_LINUX_RGBA_96_PASS",
        "candidate_cases_checked": COUNT,
        "png_sha_matches": COUNT-mismatch,
        "png_encoding_differences": mismatch,
        "rgba_sha_matches": COUNT,
        "frozen_manifest_sha256_pinned": True,
        "official_train_admitted": False,
        "official_dev_sealed": False,
        "official_ood_evaluated": False,
        "optimizer_steps": 0,
        "model_calls": 0,
        "owner_private_media_read": 0,
        "used_consumed_h173_ood": False,
    }

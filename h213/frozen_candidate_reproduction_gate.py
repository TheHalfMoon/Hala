"""H213: fail-closed bitwise verified frozen GUI research dataset source.

Only reads the self-authored, checked-in H212 research JSON and code.
Regenerates ALL 96 in-memory Pillow PNG and decoded RGBA fingerprints;
rejects any drift across host systems or changes to roles, language,
source SHA, or training-disallowing flags. NOT an admission, trainer,
model evaluation, or replacement for historical provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h212.procedural_native_gui_candidates import candidate_manifest, COUNT, SCHEMA

FROZEN_NAME = "RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json"
MAX_FROZEN_BYTES = 64 * 1024


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H213 candidate provenance gate: " + reason)


def _no_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict:
    found = {}
    for key, value in pairs:
        require(key not in found, "duplicate JSON field")
        found[key] = value
    return found


def _deny_constant(value: str):
    raise ValueError("H213 candidate provenance gate: nonstandard JSON")


def canonical_frozen_body(expected: dict) -> str:
    require(type(expected) is dict and expected.get("schema") == SCHEMA,
            "only H212 research-only metadata accepted")
    require(expected.get("official_admitted") is False
            and expected.get("dev_gold_sealed") is False
            and expected.get("official_ood_evaluated") is False
            and expected.get("gpu_optimizer_steps") == 0
            and type(expected.get("registered_candidate_count")) is int
            and expected["registered_candidate_count"] == COUNT
            and not ADMITTED_TRAIN_MANIFEST_SHAS,
            "research data cannot promote to official release")
    return json.dumps(expected, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def verify_exact_metadata_text(actual_text: str, expected_body: str) -> dict:
    """Pure expected-vs-given validation; never opens a supplied path.

    Canonical byte/character equality is deliberately stricter than
    semantic JSON equality; it rejects duplicate keys, mismatched type,
    reordered rows, formatting tricks, SHA changes and role relabeling.
    """
    require(type(actual_text) is str and type(expected_body) is str,
            "invalid frozen metadata text type")
    actual_text = actual_text.replace("\r\n", "\n")
    require(0 < len(actual_text.encode("utf-8")) <= MAX_FROZEN_BYTES,
            "frozen research metadata exceeds explicit bounds")
    try:
        parsed = json.loads(
            actual_text, object_pairs_hook=_no_duplicate_fields,
            parse_constant=_deny_constant,
        )
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("H213 candidate provenance gate: invalid JSON metadata") from None
    require(type(parsed) is dict
            and parsed.get("schema") == SCHEMA
            and parsed.get("official_admitted") is False
            and parsed.get("dev_gold_sealed") is False
            and parsed.get("official_ood_evaluated") is False
            and type(parsed.get("registered_candidate_count")) is int
            and parsed["registered_candidate_count"] == COUNT,
            "unadmitted research-only metadata identity changed")
    if actual_text != expected_body:
        # Source-created PUBLIC aggregate categories only: never emit
        # image hashes, task actions, owner media, or reference labels.
        expected = json.loads(expected_body)
        left, right = parsed.get("cases"), expected.get("cases")
        if type(left) is list and type(right) is list and len(left) == len(right):
            png = sum(type(a) is dict and a.get("image_png_sha256") != b.get("image_png_sha256")
                      for a, b in zip(left, right))
            rgba = sum(type(a) is dict and a.get("image_decoded_rgba_sha256") != b.get("image_decoded_rgba_sha256")
                       for a, b in zip(left, right))
            source = parsed.get("generator_source_sha256") != expected.get("generator_source_sha256")
            raise ValueError(
                "H213 candidate provenance gate: drift counts "
                f"png={png} rgba={rgba} generator_source_changed={source}"
            ) from None
        raise ValueError("H213 candidate provenance gate: metadata shape drift") from None
    require(actual_text == expected_body,
            "frozen GUI source image / RGBA / role / language drift")
    return {
        "status": "H213_ALL_96_GUI_RESEARCH_FINGERPRINTS_REGENERATED_IDENTICAL",
        "candidate_cases_checked": COUNT,
        "png_sha_matches": COUNT,
        "rgba_sha_matches": COUNT,
        "official_train_admitted": False,
        "official_dev_sealed": False,
        "official_ood_evaluated": False,
        "optimizer_steps": 0,
        "model_calls": 0,
        "owner_private_media_read": 0,
        "used_consumed_h173_ood": False,
    }


def verify_checked_in_source() -> dict:
    """Fixed research-only manifest path; no user-supplied source roots."""
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "H199 official TRAIN allowlist unexpectedly changed")
    p = Path(__file__).resolve().parent.parent / "h212" / FROZEN_NAME
    require(p.is_file() and not p.is_symlink(),
            "frozen research metadata missing or linked")
    require(0 < p.stat().st_size <= MAX_FROZEN_BYTES,
            "frozen research metadata oversize")
    expected = canonical_frozen_body(candidate_manifest())
    actual = p.read_text(encoding="utf-8")
    return verify_exact_metadata_text(actual, expected)


def main() -> None:
    r = verify_checked_in_source()
    print(
        r["status"],
        "cases="+str(r["candidate_cases_checked"]),
        "png="+str(r["png_sha_matches"]),
        "rgba="+str(r["rgba_sha_matches"]),
        "official_train=0 dev_sealed=0 official_ood=0",
        "private_media=0 model_calls=0 optimizer_steps=0",
    )


if __name__ == "__main__":
    main()

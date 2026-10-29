"""H188 independently audit SAVED owner-only original Pickr preaction PNGs.

This auditor never reads old screenshots/gold/model data. It may read
only the two H188 private PNGs, owner-only source input JSON and separate
H188 private source reference JSON; all stay local. Old source manifests
provide SHA-only exclusions via H185/H186, NOT old image/media bytes.
Nothing is uploaded or admitted into training.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

from PIL import Image

from h174.source_candidates import CANDIDATES
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)
from h187.pickr_preaction_native import SPECS, WIDTH, HEIGHT
from h188.owner_private_pickr_capture import (
    SCHEMA, EXPECTED_CASES, PRIVATE_DIR_NAMES,
)

TOP_FIELDS = frozenset({
    "schema", "status", "source_original_native_replayed",
    "known_old_sha_candidates_excluded",
    "complete_prior_source_history_proven",
    "complete_prior_rgba_history_proven",
    "official_train_dev_dataset_admitted",
    "case_count", "cases",
})
CASE_FIELDS = frozenset({
    "case_id", "app", "source_repo", "source_git_sha",
    "language", "instruction", "image", "image_sha256",
    "visual_rgba_sha256", "purpose", "train_eligible",
    "dev_eligible", "ood_eligible",
})
GOLD_FIELDS = frozenset({
    "case_id", "native_source_target_center",
    "not_for_model_input", "train_eligible",
})


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H188 independent owner-only source QA: " + why)


def _read_small_json(target: Path) -> dict:
    _require(target.is_file() and not target.is_symlink()
             and target.stat().st_size < 256*1024,
             "private manifest missing/unsafe/oversized")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(type(value) is dict, "private metadata object required")
    return value


def audit_owner_private_capture(
    *, owner_private_dir: Path, historical_root: Path,
) -> dict:
    _require(isinstance(owner_private_dir, Path)
             and owner_private_dir.is_absolute()
             and owner_private_dir.name in PRIVATE_DIR_NAMES
             and owner_private_dir.is_dir()
             and not owner_private_dir.is_symlink()
             and not (owner_private_dir / ".git").exists(),
             "owner-only private H188 directory required")
    _require(isinstance(historical_root, Path)
             and historical_root.is_absolute(),
             "authorized owner source SHA history root required")
    input_file = owner_private_dir / "source-prospects-not-admitted.json"
    gold_file = owner_private_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
    manifest = _read_small_json(input_file)
    reference = _read_small_json(gold_file)
    _require(
        set(manifest) == TOP_FIELDS and manifest["schema"] == SCHEMA
        and manifest["status"]
        == "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION"
        and manifest["source_original_native_replayed"] is True
        and manifest["known_old_sha_candidates_excluded"] is True
        and manifest["complete_prior_source_history_proven"] is False
        and manifest["complete_prior_rgba_history_proven"] is False
        and manifest["official_train_dev_dataset_admitted"] is False
        and manifest["case_count"] == EXPECTED_CASES,
        "source-only quarantine manifest approval/split fields changed",
    )
    rows = manifest["cases"]
    gold = reference.get("rows")
    _require(type(rows) is list and len(rows) == EXPECTED_CASES
             and type(gold) is list and len(gold) == EXPECTED_CASES
             and set(reference) == {"rows"},
             "exact quarantined private scenes and separate source gold needed")
    old_candidates = scan_legacy_manifest_sha_fields([historical_root])
    sha_pairs = []
    for i, row in enumerate(rows):
        expected = SPECS[i]
        case_id = f"h188-original-pickr-{i:02d}"
        _require(type(row) is dict and set(row) == CASE_FIELDS,
                 "source input manifest has extra/missing fields")
        _require(
            row["case_id"] == case_id and row["app"] == "pickr"
            and row["source_repo"] == CANDIDATES["pickr"]["repo"]
            and row["source_git_sha"] == CANDIDATES["pickr"]["git_sha"]
            and row["language"] == expected["language"]
            and row["instruction"] == expected["instruction"]
            and row["image"] == f"images/{case_id}.png"
            and row["purpose"] == "QUARANTINED_PROSPECTIVE_TRAIN_RESEARCH_ONLY"
            and all(row[k] is False for k in
                    ("train_eligible", "dev_eligible", "ood_eligible")),
            "original source task/role/image or admission was altered",
        )
        image_path = owner_private_dir / row["image"]
        _require(
            image_path.is_file() and not image_path.is_symlink()
            and 0 < image_path.stat().st_size < 8*1024*1024,
            "private original source PNG file missing/unsafe/oversized",
        )
        png = image_path.read_bytes()
        with Image.open(BytesIO(png)) as image:
            _require(image.format == "PNG"
                     and image.size == (WIDTH, HEIGHT),
                     "private source PNG identity or size wrong")
            rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        png_sha = sha256(png).hexdigest()
        _require(png_sha == row["image_sha256"]
                 and rgba == row["visual_rgba_sha256"],
                 "private original PNG/decoded RGBA visual SHA altered")
        source_gold = gold[i]
        _require(type(source_gold) is dict
                 and set(source_gold) == GOLD_FIELDS
                 and source_gold["case_id"] == case_id
                 and source_gold["not_for_model_input"] is True
                 and source_gold["train_eligible"] is False,
                 "private native reference gold must remain separate and inactive")
        coords = source_gold["native_source_target_center"]
        _require(type(coords) is dict and set(coords) == {"x", "y"}
                 and all(type(v) in (int, float) and math.isfinite(v)
                         for v in coords.values())
                 and 0 <= coords["x"] < WIDTH and 0 <= coords["y"] < HEIGHT,
                 "owner source-native target center invalid")
        sha_pairs.append({
            "image_sha256": png_sha, "visual_rgba_sha256": rgba,
        })
    known_sha = screen_unadmitted_new_case_hashes(sha_pairs, old_candidates)
    _require(known_sha["training_approved"] is False
             and known_sha["historical_completeness_proven"] is False,
             "known-history source reuse screen cannot authorize training")
    return {
        "status": "H188_PRIVATE_SOURCE_INPUT_PNG_AND_RGBA_INDEPENDENTLY_VERIFIED",
        "distinct_source_preaction_pngs": EXPECTED_CASES,
        "independent_original_native_owner_gold_records": EXPECTED_CASES,
        "old_owner_png_sha_candidates_compared":
            known_sha["known_legacy_png_hashes_compared"],
        "complete_prior_visual_history_proven": False,
        "train_dev_admitted": False,
        "ood_evaluation_approved": False,
        "model_calls": 0,
        "media_uploaded": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H188 owner PRIVATE PNG+RGBA SHA check, never model admission"
    )
    p.add_argument("--owner-private-dir", required=True, type=Path)
    p.add_argument("--historical-root", required=True, type=Path)
    args = p.parse_args()
    try:
        result = audit_owner_private_capture(
            owner_private_dir=args.owner_private_dir,
            historical_root=args.historical_root,
        )
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        print("H188_OWNER_PRIVATE_INDEPENDENT_QA_FAILED_CLOSED", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"{result['status']} "
        f"scenes={result['distinct_source_preaction_pngs']} "
        f"private_native_goals={result['independent_original_native_owner_gold_records']} "
        f"old_png_shas={result['old_owner_png_sha_candidates_compared']} "
        "NO_TRAIN_ADMISSION_NO_OLD_OOD_NO_MODEL_NO_UPLOAD"
    )


if __name__ == "__main__":
    main()

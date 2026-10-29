"""H190 independently verify only NEW owner-private original interact.js PNGs.

Never load any prior consumed OOD PNG, gold, old scored prediction, or
model. Reads old H185 SHA-only metadata and H188 TRAIN *manifest only*.
Owner reference drag points remain private, separate from model input.
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
from h188.owner_private_pickr_capture import _require
from h189.interact_preaction_native import SPECS, WIDTH, HEIGHT
from h190.owner_private_interact_dev_capture import (
    SCHEMA, CASES, PRIVATE_DIR_NAMES, _read_previous_train_source,
)


def _json_only(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink()
             and 0 < path.stat().st_size < 65536,
             "H190 private source metadata unsafe")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    _require(type(parsed) is dict, "H190 owner private metadata object invalid")
    return parsed


def independent_private_dev_audit(
    *, private_dir: Path, historical_root: Path, previous_train_private: Path,
) -> dict:
    _require(isinstance(private_dir, Path) and private_dir.is_absolute()
             and private_dir.name in PRIVATE_DIR_NAMES
             and private_dir.is_dir() and not private_dir.is_symlink()
             and not (private_dir / ".git").exists(),
             "H190 independent owner source quarantine missing/unsafe")
    doc = _json_only(private_dir / "source-prospects-not-admitted.json")
    ref = _json_only(private_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json")
    expected_top = {
        "schema", "status", "source_original_native_replayed",
        "known_old_sha_candidates_excluded", "different_original_source_from_train",
        "complete_prior_source_history_proven", "complete_prior_rgba_history_proven",
        "official_train_dev_dataset_admitted", "case_count", "cases",
    }
    _require(set(doc) == expected_top
             and doc["schema"] == SCHEMA
             and doc["status"] == "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION"
             and doc["source_original_native_replayed"] is True
             and doc["known_old_sha_candidates_excluded"] is True
             and doc["different_original_source_from_train"] is True
             and doc["complete_prior_source_history_proven"] is False
             and doc["complete_prior_rgba_history_proven"] is False
             and doc["official_train_dev_dataset_admitted"] is False
             and doc["case_count"] == CASES,
             "H190 frozen owner-only DEV quarantine/approval claim altered")
    rows = doc["cases"]
    gold = ref.get("rows")
    _require(type(rows) is list and len(rows) == CASES
             and set(ref) == {"rows"} and type(gold) is list and len(gold) == CASES,
             "H190 owner-private source input/gold row count incorrect")
    allowed_row = {
        "case_id", "app", "source_repo", "source_git_sha", "language",
        "instruction", "action_type", "image", "image_sha256",
        "visual_rgba_sha256", "purpose", "train_eligible", "dev_eligible",
        "ood_eligible",
    }
    allowed_goal = {
        "case_id", "private_original_drag_start", "private_original_drop_target",
        "not_for_model_input", "dev_eligible",
    }
    pairs = []
    for i, (row, private_gold) in enumerate(zip(rows, gold)):
        spec = SPECS[i]
        ident = f"h190-original-interact-{i:02d}"
        _require(type(row) is dict and set(row) == allowed_row
                 and row["case_id"] == ident
                 and row["app"] == "interact-dropzones"
                 and row["source_repo"] == CANDIDATES["interact-dropzones"]["repo"]
                 and row["source_git_sha"] == CANDIDATES["interact-dropzones"]["git_sha"]
                 and row["language"] == spec["language"]
                 and row["instruction"] == spec["instruction"]
                 and row["action_type"] == "drag"
                 and row["image"] == f"images/{ident}.png"
                 and row["purpose"] == "QUARANTINED_PROSPECTIVE_DEV_RESEARCH_ONLY"
                 and all(row[k] is False for k in
                         ("train_eligible", "dev_eligible", "ood_eligible")),
                 "H190 independent original source DEV role/task identity altered")
        img_path = private_dir / row["image"]
        _require(img_path.is_file() and not img_path.is_symlink()
                 and 0 < img_path.stat().st_size < 8*1024*1024,
                 "H190 private current source PNG missing/unsafe")
        png = img_path.read_bytes()
        with Image.open(BytesIO(png)) as image:
            _require(image.format == "PNG" and image.size == (WIDTH, HEIGHT),
                     "H190 private original PNG source size/format changed")
            rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        png_sha = sha256(png).hexdigest()
        _require(png_sha == row["image_sha256"]
                 and rgba == row["visual_rgba_sha256"],
                 "H190 private original source raster altered")
        _require(type(private_gold) is dict
                 and set(private_gold) == allowed_goal
                 and private_gold["case_id"] == ident
                 and private_gold["not_for_model_input"] is True
                 and private_gold["dev_eligible"] is False,
                 "H190 original reference drag leaked into model inputs")
        for field in ("private_original_drag_start", "private_original_drop_target"):
            pt = private_gold[field]
            _require(type(pt) is list and len(pt) == 2
                     and all(type(v) is float and math.isfinite(v)
                             for v in pt)
                     and 0 <= pt[0] < WIDTH and 0 <= pt[1] < HEIGHT,
                     "H190 original native reference drag geometry corrupted")
        pairs.append({
            "image_sha256": png_sha,
            "visual_rgba_sha256": rgba,
        })
    previous = _read_previous_train_source(previous_train_private)
    _require(
        all(p["image_sha256"] not in previous["png"]
            and p["visual_rgba_sha256"] not in previous["rgba"]
            for p in pairs),
        "H190 original DEV source overlaps private Pickr TRAIN pixels",
    )
    history = scan_legacy_manifest_sha_fields([historical_root])
    check = screen_unadmitted_new_case_hashes(pairs, history)
    _require(check["training_approved"] is False
             and check["historical_completeness_proven"] is False,
             "H190 incomplete prior source history cannot approve training")
    return {
        "status": "H190_PRIVATE_ORIGINAL_INTERACT_PREACTION_PNG_RGBA_INDEPENDENT_PASS",
        "private_preaction_scenes": CASES,
        "private_reference_drag_goals": CASES,
        "compared_old_png_sha_metadata": check["known_legacy_png_hashes_compared"],
        "cross_train_app_family_disjoint": True,
        "cross_train_visual_sha_disjoint": True,
        "train_dev_admitted": False, "complete_historical_visual_provenance": False,
        "old_ood_accessed": False, "model_calls": 0, "public_media_uploaded": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H190 owner ONLY independent new DEV preaction PNG/RGBA QA"
    )
    p.add_argument("--owner-private-dir", type=Path, required=True)
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--previous-train-private", type=Path, required=True)
    args = p.parse_args()
    try:
        result = independent_private_dev_audit(
            private_dir=args.owner_private_dir,
            historical_root=args.historical_root,
            previous_train_private=args.previous_train_private,
        )
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        print("H190_INDEPENDENT_OWNER_DEV_PREACTION_AUDIT_FAILED_CLOSED", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"{result['status']} images={result['private_preaction_scenes']} "
        f"separate_native_drag_goals={result['private_reference_drag_goals']} "
        f"old_png_sha_screened={result['compared_old_png_sha_metadata']} "
        "NO_DEV_ADMISSION_NO_MODEL_NO_OOD_NO_MEDIA_UPLOAD"
    )


if __name__ == "__main__":
    main()

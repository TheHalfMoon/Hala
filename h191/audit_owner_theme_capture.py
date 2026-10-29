"""H191 independent owner-only audit of NEW genuine Pickr theme PNGs.

Only new H191 quarantine PNGs opened; prior H188/H190 images/gold
NEVER opened. Historical H185/H186 source SHA-only manifest fields
are read to rule out known reuse. No AI/GPU/old OOD/public media.
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
from h191.pickr_three_theme_preaction import (
    CASE_COUNT, SEEDS, SPECS, THEMES, WIDTH, HEIGHT,
)
from h191.owner_private_theme_capture import (
    SCHEMA, _prior_manifest_source_metadata,
    _safe_private_directory, _safe_private_regular_file,
)

TOP_FIELDS = frozenset({
    "schema", "status", "source_original_native_replayed",
    "known_old_sha_candidates_excluded", "previous_train_dev_visual_sha_disjoint",
    "complete_prior_source_history_proven",
    "complete_prior_rgba_history_proven",
    "official_train_dev_dataset_admitted", "case_count", "cases",
})
CASE_FIELDS = frozenset({
    "case_id", "app", "source_repo", "source_git_sha", "theme",
    "seed_index", "target_index", "language", "instruction", "image",
    "image_sha256", "visual_rgba_sha256", "purpose", "train_eligible",
    "dev_eligible", "ood_eligible",
})
GOLD_FIELDS = frozenset({
    "case_id", "original_native_target_center", "not_for_model_input",
    "train_eligible",
})


def _small_json(path: Path) -> dict:
    _require(_safe_private_regular_file(path)
             and 0 < path.stat().st_size < 256*1024,
             "H191 saved owner-private metadata missing or unsafe")
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(type(result) is dict, "H191 metadata object required")
    return result


def audit_saved_owner_theme_preactions(
    *, new_private_dir: Path, previous_pickr_private: Path,
    previous_dev_private: Path, history_root: Path,
) -> dict:
    _require(isinstance(new_private_dir, Path)
             and new_private_dir.name == "Hala-H191-Private"
             and _safe_private_directory(new_private_dir)
             and _safe_private_directory(new_private_dir / "images"),
             "H191 actual original private source folder required")
    doc = _small_json(new_private_dir / "source-prospects-not-admitted.json")
    refs = _small_json(
        new_private_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
    )
    _require(set(doc) == TOP_FIELDS
             and doc["schema"] == SCHEMA
             and doc["status"] == "H191_TWELVE_PRIVATE_PICKR_PREACTION_QUARANTINE_ONLY"
             and doc["source_original_native_replayed"] is True
             and doc["known_old_sha_candidates_excluded"] is True
             and doc["previous_train_dev_visual_sha_disjoint"] is True
             and doc["complete_prior_source_history_proven"] is False
             and doc["complete_prior_rgba_history_proven"] is False
             and doc["official_train_dev_dataset_admitted"] is False
             and type(doc["case_count"]) is int
             and doc["case_count"] == CASE_COUNT,
             "H191 private case admission/source history claim changed")
    rows = doc["cases"]
    goals = refs.get("rows")
    _require(type(rows) is list and len(rows) == CASE_COUNT
             and set(refs) == {"rows"}
             and type(goals) is list and len(goals) == CASE_COUNT,
             "H191 saved private image/reference count changed")
    pair_hashes = []
    for i, (spec, row, goal) in enumerate(zip(SPECS, rows, goals)):
        case_id = f"h191-original-pickr-{i:02d}"
        _require(type(row) is dict and set(row) == CASE_FIELDS
                 and row["case_id"] == case_id
                 and row["app"] == "pickr"
                 and row["source_repo"] == CANDIDATES["pickr"]["repo"]
                 and row["source_git_sha"] == CANDIDATES["pickr"]["git_sha"]
                 and row["theme"] == spec["theme"]
                 and row["seed_index"] == spec["seed_index"]
                 and row["target_index"] == spec["target_index"]
                 and row["language"] == spec["language"]
                 and row["instruction"] == spec["instruction"]
                 and row["image"] == f"images/{case_id}.png"
                 and row["purpose"] == "QUARANTINED_ORIGINAL_THREE_THEME_TRAIN_RESEARCH_ONLY"
                 and all(row[k] is False for k in
                         ("train_eligible", "dev_eligible", "ood_eligible")),
                 "H191 original source task/data role identity modified")
        filename = new_private_dir / row["image"]
        _require(_safe_private_regular_file(filename)
                 and 0 < filename.stat().st_size < 8*1024*1024,
                 "H191 owner saved source PNG missing or oversized")
        png = filename.read_bytes()
        with Image.open(BytesIO(png)) as img:
            _require(img.format == "PNG" and img.size == (WIDTH, HEIGHT),
                     "H191 saved source screenshot format/viewport drift")
            rgba = sha256(img.convert("RGBA").tobytes()).hexdigest()
        png_sha = sha256(png).hexdigest()
        _require(png_sha == row["image_sha256"]
                 and rgba == row["visual_rgba_sha256"],
                 "H191 saved source PNG/decoded RGBA corruption")
        _require(type(goal) is dict and set(goal) == GOLD_FIELDS
                 and goal["case_id"] == case_id
                 and goal["not_for_model_input"] is True
                 and goal["train_eligible"] is False,
                 "H191 source-native reference gold leaked into model input")
        xy = goal["original_native_target_center"]
        _require(type(xy) is dict and set(xy) == {"x", "y"}
                 and all(type(v) is float and math.isfinite(v)
                         for v in xy.values())
                 and 0 <= xy["x"] < WIDTH and 0 <= xy["y"] < HEIGHT,
                 "H191 private source-native target coordinate corrupted")
        pair_hashes.append({
            "image_sha256": png_sha, "visual_rgba_sha256": rgba,
        })
    older_pickr = _prior_manifest_source_metadata(
        previous_pickr_private, "Hala-H188-Private", "pickr",
        "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    older_interact = _prior_manifest_source_metadata(
        previous_dev_private, "Hala-H190-Private", "interact-dropzones",
        "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    for candidate in pair_hashes:
        for prior in (older_pickr, older_interact):
            _require(candidate["image_sha256"] not in prior["png"]
                     and candidate["visual_rgba_sha256"] not in prior["rgba"],
                     "H191 source pixels duplicate previous TRAIN or DEV")
    old = scan_legacy_manifest_sha_fields([history_root])
    screen_unadmitted_new_case_hashes(pair_hashes, old)
    return {
        "status": "H191_INDEPENDENT_SAVED_ORIGINAL_PICKR_PNG_RGBA_SHA_PASS",
        "actual_new_images_rehashed": CASE_COUNT,
        "actual_native_owner_gold_separately_checked": CASE_COUNT,
        "source_ui_themes": len(THEMES),
        "source_native_start_colors": len(SEEDS),
        "old_png_sha_exclusions": len(old["_owner_only_png_sha_candidates"]),
        "source_train_dev_visual_disjoint": True,
        "train_dev_admitted": False,
        "historical_visual_history_complete": False,
        "ood_accessed": False,
        "model_calls": 0,
        "uploaded_images": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H191 owner-only source PNG and decoded RGBA check; never data admission"
    )
    p.add_argument("--owner-private-dir", type=Path, required=True)
    p.add_argument("--prior-pickr-private", type=Path, required=True)
    p.add_argument("--prior-dev-private", type=Path, required=True)
    p.add_argument("--historical-root", type=Path, required=True)
    args = p.parse_args()
    try:
        result = audit_saved_owner_theme_preactions(
            new_private_dir=args.owner_private_dir,
            previous_pickr_private=args.prior_pickr_private,
            previous_dev_private=args.prior_dev_private,
            history_root=args.historical_root,
        )
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError):
        print("H191_OWNER_PRIVATE_SOURCE_AUDIT_FAILED_CLOSED", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"{result['status']} "
        f"real_private_png={result['actual_new_images_rehashed']} "
        f"original_native_private_gold={result['actual_native_owner_gold_separately_checked']} "
        f"known_prior_png_sha={result['old_png_sha_exclusions']} "
        f"source_themes={result['source_ui_themes']} "
        "NO_ADMISSION_NO_MODEL_NO_OLD_OOD_NO_UPLOAD"
    )


if __name__ == "__main__":
    main()

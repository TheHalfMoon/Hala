"""H194 independently audit ONLY the two new owner-private snap slider PNGs.

Never open previous H188/H190/H191 media/gold or consumed H173 OOD.
Old source manifests contribute SHA-only historical exclusions. New
owner reference actions stay separate from model-visible inputs.
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
from h193.interact_snap_preaction import SPECS, WIDTH, HEIGHT
from h194.owner_private_snap_capture import (
    SCHEMA, STATUS, SOURCE, PREVIOUS, _safe_lexical_path,
    _prior_visual_sha_only,
)

CASE_FIELDS = frozenset({
    "case_id", "app", "source_repo", "source_git_sha",
    "language", "instruction", "target_slider", "action_type",
    "image", "image_sha256", "visual_rgba_sha256", "purpose",
    "train_eligible", "dev_eligible", "ood_eligible",
})
HEADER_FIELDS = frozenset({
    "schema", "status", "source_original_native_replayed",
    "all_known_old_shas_screened", "prior_train_dev_visual_disjoint",
    "historical_source_complete", "historical_decoded_rgba_complete",
    "official_train_dev_dataset_admitted", "case_count", "cases",
})
REF_FIELDS = frozenset({
    "case_id", "private_original_action",
    "not_for_model_input", "dev_eligible",
})


def _small_private_json(path: Path) -> dict:
    _safe_lexical_path(path)
    _require(path.is_file() and 0 < path.stat().st_size < 256*1024,
             "unsafe owner-private source manifest/reference")
    obj = json.loads(path.read_text(encoding="utf-8"))
    _require(type(obj) is dict, "owner-private metadata must be an object")
    return obj


def audit_owner_new_slider(
    *, private_dir: Path, older_private_root: Path, history_root: Path,
) -> dict:
    _safe_lexical_path(private_dir)
    _require(private_dir.name == "Hala-H194-Private"
             and private_dir.is_dir()
             and not (private_dir / ".git").exists(),
             "H194 owner-private newly captured original source only")
    doc = _small_private_json(private_dir / "source-prospects-not-admitted.json")
    reference = _small_private_json(
        private_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
    )
    _require(set(doc) == HEADER_FIELDS
             and doc["schema"] == SCHEMA and doc["status"] == STATUS
             and doc["source_original_native_replayed"] is True
             and doc["all_known_old_shas_screened"] is True
             and doc["prior_train_dev_visual_disjoint"] is True
             and doc["historical_source_complete"] is False
             and doc["historical_decoded_rgba_complete"] is False
             and doc["official_train_dev_dataset_admitted"] is False
             and type(doc["case_count"]) is int
             and doc["case_count"] == len(SPECS),
             "H194 false historical completeness or data admission")
    rows = doc["cases"]
    gold = reference.get("rows")
    _require(type(rows) is list and len(rows) == len(SPECS)
             and set(reference) == {"rows"}
             and type(gold) is list and len(gold) == len(SPECS),
             "H194 original source cases and private reference separation broken")
    pairs = []
    for index, (spec, row, ref) in enumerate(zip(SPECS, rows, gold)):
        case_id = f"h194-original-interact-snap-{index:02d}"
        _require(type(row) is dict and set(row) == CASE_FIELDS
                 and row["case_id"] == case_id
                 and row["app"] == "interact-snap"
                 and row["source_repo"] == CANDIDATES[SOURCE]["repo"]
                 and row["source_git_sha"] == CANDIDATES[SOURCE]["git_sha"]
                 and row["language"] == spec["language"]
                 and row["instruction"] == spec["instruction"]
                 and row["target_slider"] == spec["target"]
                 and row["action_type"] == "key_press"
                 and row["image"] == f"images/{case_id}.png"
                 and row["purpose"]
                     == "QUARANTINED_PROSPECTIVE_DEV_SLIDER_RESEARCH_ONLY"
                 and all(row[k] is False for k in
                         ("train_eligible", "dev_eligible", "ood_eligible")),
                 "H194 immutable original task/model input role changed")
        file = private_dir / row["image"]
        _safe_lexical_path(file)
        _require(file.is_file() and 0 < file.stat().st_size < 8*1024*1024,
                 "unsafe/missing newly captured source PNG")
        png = file.read_bytes()
        with Image.open(BytesIO(png)) as img:
            _require(img.format == "PNG" and img.size == (WIDTH, HEIGHT),
                     "new original screenshot not a 960x720 PNG")
            rgba = sha256(img.convert("RGBA").tobytes()).hexdigest()
        png_sha = sha256(png).hexdigest()
        _require(row["image_sha256"] == png_sha
                 and row["visual_rgba_sha256"] == rgba,
                 "H194 saved original PNG or decoded RGBA changed")
        _require(type(ref) is dict and set(ref) == REF_FIELDS
                 and ref["case_id"] == case_id
                 and ref["not_for_model_input"] is True
                 and ref["dev_eligible"] is False,
                 "H194 native private key/goal leaked into model input")
        goal = ref["private_original_action"]
        _require(type(goal) is dict
                 and set(goal) == {
                     "action_type", "key", "slider", "target_center"
                 }
                 and goal["action_type"] == "key_press"
                 and goal["key"] == "ArrowRight"
                 and goal["slider"] == spec["target"],
                 "H194 true native slider keyboard reference altered")
        point = goal["target_center"]
        _require(type(point) is list and len(point) == 2
                 and all(type(v) is float and math.isfinite(v)
                         for v in point)
                 and 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT,
                 "H194 original native slider reference point corrupted")
        pairs.append({
            "image_sha256": png_sha, "visual_rgba_sha256": rgba,
        })
    _require(len({x["image_sha256"] for x in pairs}) == len(SPECS)
             and len({x["visual_rgba_sha256"] for x in pairs}) == len(SPECS),
             "H194 duplicate newly captured original screenshot pixels")

    prior = [
        _prior_visual_sha_only(
            older_private_root / name, status, count, family
        )
        for name, status, count, family in PREVIOUS
    ]
    _require(all(
        item["image_sha256"] not in earlier["png"]
        and item["visual_rgba_sha256"] not in earlier["rgba"]
        for item in pairs for earlier in prior
    ), "H194 original slider inputs overlap prior train/dev sources")
    history = scan_legacy_manifest_sha_fields([history_root])
    approval = screen_unadmitted_new_case_hashes(pairs, history)
    _require(approval["training_approved"] is False
             and approval["historical_completeness_proven"] is False,
             "H194 incomplete history cannot approve training")
    return {
        "status": "H194_INDEPENDENT_PRIVATE_SNAP_ORIGINAL_PNG_RGBA_PASS",
        "new_private_preaction_images_rehashed": len(pairs),
        "private_native_keyboard_reference_records": len(gold),
        "previous_new_source_images_compared": sum(x[2] for x in PREVIOUS),
        "old_png_sha_candidates_compared": len(history["_owner_only_png_sha_candidates"]),
        "train_dev_admitted": False,
        "full_historical_source_and_visual_provenance": False,
        "old_ood_accessed": False,
        "model_calls": 0,
        "public_images_uploaded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H194 independent owner new snap slider PNG/RGBA check, NO dataset"
    )
    parser.add_argument("--owner-private-dir", required=True, type=Path)
    parser.add_argument("--prior-private-root", required=True, type=Path)
    parser.add_argument("--historical-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit_owner_new_slider(
            private_dir=args.owner_private_dir,
            older_private_root=args.prior_private_root,
            history_root=args.historical_root,
        )
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        print("H194_PRIVATE_NEW_SOURCE_AUDIT_FAILED_CLOSED", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"{result['status']} "
        f"saved_png_rgba={result['new_private_preaction_images_rehashed']} "
        f"separate_native_actions={result['private_native_keyboard_reference_records']} "
        f"prior_private_visuals={result['previous_new_source_images_compared']} "
        f"old_png_sha={result['old_png_sha_candidates_compared']} "
        "NO_DATA_ADMISSION_NO_OLD_OOD_NO_MODEL_NO_UPLOAD"
    )


if __name__ == "__main__":
    main()

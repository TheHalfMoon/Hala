"""H191 private-only twelve real Pickr PRE-action TRAIN research prospects.

Owner-local new media only; no training admission, no OOD, no uploads.
Original app/native pixel replay and 2,677 known old SHA exclusions
MUST pass before writing ANY private source media. H188 and H190
source manifests are read ONLY as SHA metadata: no previous images,
private targets or prior OOD reference gold are opened.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path

from PIL import Image

from h174.source_candidates import CANDIDATES, verify_local_source_only
from h188.owner_private_pickr_capture import _require
from h191.pickr_three_theme_preaction import (
    CASE_COUNT, REPEATS, SPECS, WIDTH, HEIGHT,
    run_original_theme_batch,
)

SCHEMA = "hala.h191.original-pickr-three-theme-private-quarantine.v1"


def _safe_private_directory(path: Path) -> bool:
    """Reject junctions/symlinks in every owner-private ancestor, including Git."""
    if not isinstance(path, Path) or not path.is_absolute():
        return False
    try:
        for part in (path, *path.parents):
            if (not part.is_dir() or part.is_symlink()
                    or (getattr(part.lstat(), "st_file_attributes", 0) & 0x400)
                    or (part / ".git").exists()
                    or (part / ".git").is_symlink()):
                return False
    except OSError:
        return False
    return True


def _safe_private_regular_file(path: Path) -> bool:
    """Reject Windows reparse files as well as ordinary symlinks."""
    try:
        return (path.is_file() and not path.is_symlink()
                and not (getattr(path.lstat(), "st_file_attributes", 0) & 0x400))
    except OSError:
        return False


def _validate_new_private_folder(dest: Path) -> None:
    _require(isinstance(dest, Path) and dest.is_absolute()
             and dest.name == "Hala-H191-Private",
             "H191 requires exact new absolute owner-private output")
    _require(not os.path.lexists(dest),
             "H191 refuses overwrite of existing owner-private images")
    for part in dest.parents:
        _require(part.is_dir() and not part.is_symlink()
                 and not (getattr(part.lstat(), "st_file_attributes", 0) & 0x400)
                 and not (part / ".git").exists()
                 and not (part / ".git").is_symlink(),
                 "H191 private output must stay outside Git or junctions")


def _prior_manifest_source_metadata(
    owner_private: Path, expected_name: str, source_key: str,
    expected_status: str,
) -> dict:
    _require(isinstance(owner_private, Path)
             and owner_private.name == expected_name
             and _safe_private_directory(owner_private),
             "H191 pinned previous source quarantine directory unavailable")
    p = owner_private / "source-prospects-not-admitted.json"
    _require(_safe_private_regular_file(p)
             and 0 < p.stat().st_size < 65536,
             "H191 previous source-only visual metadata missing/oversized")
    doc = json.loads(p.read_text(encoding="utf-8"))
    _require(type(doc) is dict
             and doc.get("status") == expected_status
             and doc.get("official_train_dev_dataset_admitted") is False
             and type(doc.get("cases")) is list
             and len(doc["cases"]) == 2,
             "H191 previous source split is not quarantined")
    pins, png, rgba = set(), set(), set()
    expected_pin = CANDIDATES[source_key]
    for row in doc["cases"]:
        _require(type(row) is dict
                 and all(row.get(k) is False
                         for k in ("train_eligible", "dev_eligible", "ood_eligible"))
                 and row.get("source_repo") == expected_pin["repo"]
                 and row.get("source_git_sha") == expected_pin["git_sha"],
                 "H191 original source previous quarantine identity changed")
        for key in ("image_sha256", "visual_rgba_sha256"):
            value = row.get(key)
            _require(type(value) is str and len(value) == 64
                     and all(c in "0123456789abcdef" for c in value),
                     "H191 previous source visual digest malformed")
        pins.add((row["source_repo"], row["source_git_sha"]))
        png.add(row["image_sha256"])
        rgba.add(row["visual_rgba_sha256"])
    _require(len(png) == len(rgba) == 2 and len(pins) == 1,
             "H191 previous source PNG/RGBA duplicates")
    return {"pins": pins, "png": png, "rgba": rgba}


def _verified_new_preaction_memory(captures: list[tuple]) -> list[dict]:
    _require(type(captures) is list
             and len(captures) == CASE_COUNT * REPEATS,
             "H191 incomplete original browser preaction source memory")
    accepted = []
    for spec in SPECS:
        rows = [row for row in captures if row[0] == spec]
        _require(len(rows) == REPEATS,
                 "H191 missing independent original source preaction replay")
        (_, png, goal), (_, other_png, other_goal) = rows
        _require(type(png) is bytes and len(png) > 100
                 and png == other_png and goal == other_goal,
                 "H191 independent source pixel or target center changed")
        _require(type(goal) is dict and set(goal) == {"x", "y"}
                 and all(type(v) is float and math.isfinite(v)
                         for v in goal.values())
                 and 0 <= goal["x"] < WIDTH and 0 <= goal["y"] < HEIGHT,
                 "H191 native source target center invalid")
        with Image.open(BytesIO(png)) as image:
            _require(image.format == "PNG" and image.size == (WIDTH, HEIGHT),
                     "H191 source preaction PNG format or size invalid")
            rgba_sha = sha256(image.convert("RGBA").tobytes()).hexdigest()
        accepted.append({
            "spec": spec, "png": png, "goal": goal,
            "png_sha256": sha256(png).hexdigest(),
            "rgba_sha256": rgba_sha,
        })
    _require(len({r["png_sha256"] for r in accepted}) == CASE_COUNT
             and len({r["rgba_sha256"] for r in accepted}) == CASE_COUNT,
             "H191 source preaction pixel duplicates")
    return accepted


def quarantine_private_original_theme_source(
    *, original_donor: Path, source_built_dist: Path,
    owner_history_root: Path, prior_pickr_private: Path,
    prior_dev_private: Path, new_private_dir: Path,
    browser_channel: str,
) -> dict:
    """Write only NEW H191 Pickr images after complete native source+SHA QA."""
    _validate_new_private_folder(new_private_dir)
    _require(isinstance(original_donor, Path)
             and isinstance(source_built_dist, Path)
             and isinstance(owner_history_root, Path)
             and all(p.is_absolute() for p in
                     (original_donor, source_built_dist, owner_history_root)),
             "H191 original Git, scratch and owner old SHA roots required")
    pin = verify_local_source_only("pickr", original_donor)
    _require(pin["source_git_sha"] == CANDIDATES["pickr"]["git_sha"],
             "H191 original donor immutable Git pin changed")
    scratch = source_built_dist.parent
    for rel in (CANDIDATES["pickr"]["html_path"],
                CANDIDATES["pickr"]["license_path"],
                "www/index.js"):
        a, b = original_donor / rel, scratch / rel
        _require(a.is_file() and b.is_file()
                 and sha256(a.read_bytes()).digest()
                 == sha256(b.read_bytes()).digest(),
                 "H191 original demo source scratch provenance mismatch")
    old_train = _prior_manifest_source_metadata(
        prior_pickr_private, "Hala-H188-Private", "pickr",
        "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    old_dev = _prior_manifest_source_metadata(
        prior_dev_private, "Hala-H190-Private", "interact-dropzones",
        "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
    )
    _require(old_train["pins"].isdisjoint(old_dev["pins"]),
             "H191 TRAIN and DEV original implementation families overlap")
    captures: list[tuple] = []
    proof = run_original_theme_batch(
        source_built_dist, browser_channel, owner_history_root,
        capture=lambda spec, png, goal: captures.append(
            (dict(spec), bytes(png), dict(goal))
        ),
    )
    _require(proof["unique_original_preaction_scenes"] == CASE_COUNT
             and proof["original_native_seed_and_target_actions"]
             == CASE_COUNT * REPEATS * 2
             and proof["owner_old_image_sha_screened"] is True
             and proof["official_train_cases_admitted"] == 0
             and proof["model_calls"] == 0,
             "H191 actual source-native or owner historical SHA gate failed")
    verified = _verified_new_preaction_memory(captures)
    for new in verified:
        for older in (old_train, old_dev):
            _require(new["png_sha256"] not in older["png"]
                     and new["rgba_sha256"] not in older["rgba"],
                     "H191 source image repeats a previous TRAIN/DEV cohort")
    # After ALL original source native actions and collision checks pass:
    new_private_dir.mkdir(mode=0o700, exist_ok=False)
    (new_private_dir / "images").mkdir(mode=0o700)
    cases, reference = [], []
    for i, row in enumerate(verified):
        spec = row["spec"]
        case_id = f"h191-original-pickr-{i:02d}"
        image_path = f"images/{case_id}.png"
        (new_private_dir / image_path).write_bytes(row["png"])
        cases.append({
            "case_id": case_id,
            "app": "pickr",
            "source_repo": CANDIDATES["pickr"]["repo"],
            "source_git_sha": CANDIDATES["pickr"]["git_sha"],
            "theme": spec["theme"],
            "seed_index": spec["seed_index"],
            "target_index": spec["target_index"],
            "language": spec["language"],
            "instruction": spec["instruction"],
            "image": image_path,
            "image_sha256": row["png_sha256"],
            "visual_rgba_sha256": row["rgba_sha256"],
            "purpose": "QUARANTINED_ORIGINAL_THREE_THEME_TRAIN_RESEARCH_ONLY",
            "train_eligible": False, "dev_eligible": False,
            "ood_eligible": False,
        })
        reference.append({
            "case_id": case_id,
            "original_native_target_center": row["goal"],
            "not_for_model_input": True,
            "train_eligible": False,
        })
    (new_private_dir / "source-prospects-not-admitted.json").write_text(
        json.dumps({
            "schema": SCHEMA,
            "status": "H191_TWELVE_PRIVATE_PICKR_PREACTION_QUARANTINE_ONLY",
            "source_original_native_replayed": True,
            "known_old_sha_candidates_excluded": True,
            "previous_train_dev_visual_sha_disjoint": True,
            "complete_prior_source_history_proven": False,
            "complete_prior_rgba_history_proven": False,
            "official_train_dev_dataset_admitted": False,
            "case_count": len(cases),
            "cases": cases,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (new_private_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": reference}, indent=2) + "\n", encoding="utf-8",
    )
    return {
        "status": "H191_REAL_TWELVE_ORIGINAL_PICKR_PRIVATE_PREACTION_SCENES_QUARANTINED",
        "new_preaction_pngs": len(cases),
        "new_private_source_goals": len(reference),
        "real_source_native_actions": proof["original_native_seed_and_target_actions"],
        "distinct_themes": len({x["theme"] for x in cases}),
        "distinct_seed_colors": len({x["seed_index"] for x in cases}),
        "owner_old_shas_checked": True,
        "cross_train_dev_shas_disjoint": True,
        "official_train_dev_admitted": False,
        "new_ood_images": 0, "model_calls": 0, "gpu_steps": 0,
        "uploaded_media": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H191 owner-private genuine 12 original-theme preaction PNGs (no TRAIN admission)"
    )
    p.add_argument("--donor-root", required=True, type=Path)
    p.add_argument("--built-dist", required=True, type=Path)
    p.add_argument("--historical-root", required=True, type=Path)
    p.add_argument("--prior-pickr-private", required=True, type=Path)
    p.add_argument("--prior-dev-private", required=True, type=Path)
    p.add_argument("--owner-private-dir", required=True, type=Path)
    p.add_argument("--browser-channel", default="msedge")
    args = p.parse_args()
    proof = quarantine_private_original_theme_source(
        original_donor=args.donor_root, source_built_dist=args.built_dist,
        owner_history_root=args.historical_root,
        prior_pickr_private=args.prior_pickr_private,
        prior_dev_private=args.prior_dev_private,
        new_private_dir=args.owner_private_dir,
        browser_channel=args.browser_channel,
    )
    print(
        f"{proof['status']} "
        f"new_private_png={proof['new_preaction_pngs']} "
        f"real_native_actions={proof['real_source_native_actions']} "
        f"original_themes={proof['distinct_themes']} "
        f"original_seed_colors={proof['distinct_seed_colors']} "
        "NO_DATA_ADMISSION_NO_OOD_NO_GPU_NO_PUBLIC_MEDIA"
    )


if __name__ == "__main__":
    main()

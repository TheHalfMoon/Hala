"""H188 private-only quarantine of REAL original preaction source PNGs.

No fake generated media, no evaluation/consumed OOD, no GPU and NO dataset
admission. Captures the actual H187 original Pickr PRE-action PNG plus native
swatch target center into a NEW, non-repository, owner-private folder only
AFTER source replay succeeds and the known owner historical SHA exclusions
have been checked. Real screenshots & gold NEVER enter the public Git repo.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat

from h174.source_candidates import CANDIDATES
from h187.pickr_preaction_native import (
    SPECS, REPEATS, WIDTH, HEIGHT, run_source_only,
)

SCHEMA = "hala.h188.owner-private-new-preaction-quarantine.v1"
PRIVATE_DIR_NAMES = frozenset({"Hala-H188-Private", "private-h188"})
EXPECTED_CASES = 2


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H188 owner-private genuine GUI source: " + reason)


def _private_destination(path: Path) -> Path:
    """Require a brand new owner-private destination, not inside Git tree."""
    _require(isinstance(path, Path) and path.is_absolute(),
             "absolute owner-private location required")
    _require(path.name in PRIVATE_DIR_NAMES,
             "owner-private H188 filename required")
    _require(not os.path.lexists(path),
             "refusing overwrite of prior private source images")
    for ancestor in path.parents:
        _require(ancestor.is_dir() and not ancestor.is_symlink(),
                 "unsafe owner destination parent")
        attrs = getattr(ancestor.lstat(), "st_file_attributes", 0)
        _require(not (attrs & 0x400), "owner destination junction/reparse")
        _require(not (ancestor / ".git").exists()
                 and not (ancestor / ".git").is_symlink(),
                 "never write native source photos within a Git worktree")
    return path


def _verify_ephemeral_preview(captures: list[tuple[dict, bytes, dict]]) -> list[dict]:
    """Conservatively validate real before-action PNGs and owner-only goals."""
    from PIL import Image

    _require(type(captures) is list and len(captures) == EXPECTED_CASES*REPEATS,
             "original fresh browser source capture replay incomplete")
    ordered: list[dict] = []
    for spec in SPECS:
        originals = [row for row in captures if row[0] == spec]
        _require(len(originals) == REPEATS,
                 "exact two native source captures required per scenario")
        (meta, png, point), (again_meta, again_png, again_point) = originals
        _require(type(png) is bytes and len(png) > 100
                 and png == again_png and point == again_point,
                 "preaction PNG or original native target geometry drift")
        _require(type(point) is dict and set(point) == {"x", "y"}
                 and all(type(v) in (int, float) and math.isfinite(v)
                         for v in point.values())
                 and 0 <= point["x"] < WIDTH and 0 <= point["y"] < HEIGHT,
                 "original target coordinate invalid")
        with Image.open(BytesIO(png)) as image:
            _require(image.format == "PNG" and image.size == (WIDTH, HEIGHT),
                     "original preaction image is not 960x720 PNG")
            rgba_sha = sha256(image.convert("RGBA").tobytes()).hexdigest()
        ordered.append({
            "spec": spec, "png": png, "goal": point,
            "png_sha256": sha256(png).hexdigest(),
            "rgba_sha256": rgba_sha,
        })
    _require(len({r["png_sha256"] for r in ordered}) == EXPECTED_CASES
             and len({r["rgba_sha256"] for r in ordered}) == EXPECTED_CASES,
             "two training prospect input screenshots not actually different")
    return ordered


def quarantine_owner_source_only(
    *, dist_dir: Path, browser_channel: str, historical_root: Path,
    owner_private_dir: Path,
) -> dict:
    """Only actual browser runner can create owner-private quarantined images."""
    _require(type(historical_root) is Path and historical_root.is_absolute(),
             "real owner history root required for known old SHA exclusions")
    dest = _private_destination(owner_private_dir)
    captures: list[tuple[dict, bytes, dict]] = []
    # First original source-native action proof, two context replay,
    # and OWNER historical 2,677-known-old SHA exclusion -- NO WRITES.
    proof = run_source_only(
        dist_dir, browser_channel, historical_root,
        capture_preaction=lambda spec, png, point: captures.append(
            (dict(spec), bytes(png), dict(point))
        ),
    )
    _require(proof["original_preaction_scenes"] == EXPECTED_CASES
             and proof["known_owner_history_shas_screened"] is True
             and proof["train_dev_examples_admitted"] == 0
             and proof["model_training_approved"] is False,
             "owner source replay did not pass history exclusion")
    originals = _verify_ephemeral_preview(captures)
    # Only after every independent native proof passes can any media
    # be written, and only outside all Git worktrees.
    dest.mkdir(mode=0o700, parents=False, exist_ok=False)
    media = dest / "images"
    media.mkdir(mode=0o700)
    input_rows: list[dict] = []
    reference_rows: list[dict] = []
    for i, row in enumerate(originals):
        case_id = f"h188-original-pickr-{i:02d}"
        rel = f"images/{case_id}.png"
        (dest / rel).write_bytes(row["png"])
        spec = row["spec"]
        input_rows.append({
            "case_id": case_id,
            "app": "pickr",
            "source_repo": CANDIDATES["pickr"]["repo"],
            "source_git_sha": CANDIDATES["pickr"]["git_sha"],
            "language": spec["language"],
            "instruction": spec["instruction"],
            "image": rel,
            "image_sha256": row["png_sha256"],
            "visual_rgba_sha256": row["rgba_sha256"],
            "purpose": "QUARANTINED_PROSPECTIVE_TRAIN_RESEARCH_ONLY",
            "train_eligible": False,
            "dev_eligible": False,
            "ood_eligible": False,
        })
        reference_rows.append({
            "case_id": case_id, "native_source_target_center": row["goal"],
            "not_for_model_input": True,
            "train_eligible": False,
        })
    doc = {
        "schema": SCHEMA,
        "status": "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
        "source_original_native_replayed": True,
        "known_old_sha_candidates_excluded": True,
        "complete_prior_source_history_proven": False,
        "complete_prior_rgba_history_proven": False,
        "official_train_dev_dataset_admitted": False,
        "case_count": len(input_rows),
        "cases": input_rows,
    }
    # Owner-only visual task inputs and source gold are separate. No
    # public CI action ever calls this function, and no upload is made.
    (dest / "source-prospects-not-admitted.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8",
    )
    (dest / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": reference_rows}, indent=2) + "\n", "utf-8",
    )
    return {
        "status": "H188_TWO_PRIVATE_ORIGINAL_PREACTION_SCENES_QUARANTINED_ONLY",
        "owner_private_images_written": EXPECTED_CASES,
        "owner_private_native_reference_goals_written": EXPECTED_CASES,
        "train_dev_admitted": False,
        "source_history_complete": False,
        "old_rgba_history_complete": False,
        "model_calls": 0,
        "gpu_optimizer_steps": 0,
        "public_images_uploaded": 0,
        "public_gold_uploaded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H188 owner-private PRE-action original source images; NO TRAIN ADMISSION"
    )
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--browser-channel", default="chromium")
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--owner-private-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = quarantine_owner_source_only(
        dist_dir=args.dist_dir, browser_channel=args.browser_channel,
        historical_root=args.historical_root,
        owner_private_dir=args.owner_private_dir,
    )
    print(
        f"{receipt['status']} "
        f"images={receipt['owner_private_images_written']} "
        f"owner_reference_goals={receipt['owner_private_native_reference_goals_written']} "
        "NO_TRAIN_ADMISSION_NO_GPU_NO_OOD_NO_PUBLIC_MEDIA"
    )


if __name__ == "__main__":
    main()

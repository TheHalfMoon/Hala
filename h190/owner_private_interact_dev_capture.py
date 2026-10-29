"""H190 owner-private interact.js genuine PREACTION DEV research capture.

Copy-first H189 native seeded interactions and H188 quarantine pattern.
Never publish originals, private gold, old screenshots, or scored OOD.
Two preaction native frames; source owner original SHA-only exclusion;
TRAIN Pickr source-family and visual SHA disjoint; no actual DEV admission.
"""
from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
from threading import Thread

from PIL import Image

from h174.source_candidates import CANDIDATES, verify_local_source_only
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h188.owner_private_pickr_capture import _require
from h189.interact_preaction_native import (
    SPECS, WIDTH, HEIGHT, REPEATS, run_native_preaction,
)

SCHEMA = "hala.h190.owner-private-interact-dev-preaction-quarantine.v1"
PRIVATE_DIR_NAMES = frozenset({"Hala-H190-Private", "private-h190"})
SOURCE_ID = "interact-dropzones"
CASES = 2


def private_output_destination(path: Path) -> Path:
    _require(isinstance(path, Path) and path.is_absolute()
             and path.name in PRIVATE_DIR_NAMES,
             "H190 absolute named owner-private output required")
    _require(not os.path.lexists(path),
             "H190 must not overwrite another private source capture")
    for parent in path.parents:
        _require(parent.is_dir() and not parent.is_symlink(),
                 "H190 destination parent unsafe or missing")
        attr = getattr(parent.lstat(), "st_file_attributes", 0)
        _require(not (attr & 0x400)
                 and not (parent / ".git").exists()
                 and not (parent / ".git").is_symlink(),
                 "H190 must not write inside a Git checkout or junction")
    return path


def _read_previous_train_source(path: Path) -> dict:
    """Owner-private H188 manifest only, NO image or private gold reads."""
    _require(isinstance(path, Path) and path.is_absolute()
             and path.name == "Hala-H188-Private" and path.is_dir(),
             "H190 requires original private prospective TRAIN source")
    target = path / "source-prospects-not-admitted.json"
    _require(target.is_file() and not target.is_symlink()
             and target.stat().st_size < 65536,
             "H190 earlier private TRAIN SHA metadata unavailable")
    doc = json.loads(target.read_text(encoding="utf-8"))
    _require(type(doc) is dict
             and doc.get("status") == "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION"
             and doc.get("official_train_dev_dataset_admitted") is False
             and type(doc.get("cases")) is list
             and len(doc["cases"]) == CASES,
             "H190 prospective train source is not a frozen quarantine")
    png, rgba, families = set(), set(), set()
    for row in doc["cases"]:
        _require(type(row) is dict and row.get("train_eligible") is False
                 and row.get("dev_eligible") is False
                 and row.get("ood_eligible") is False,
                 "H190 cross-split previous prospect eligibility drift")
        _require(type(row.get("image_sha256")) is str
                 and type(row.get("visual_rgba_sha256")) is str,
                 "H190 prior TRAIN visual SHA metadata malformed")
        png.add(row["image_sha256"])
        rgba.add(row["visual_rgba_sha256"])
        families.add((row.get("source_repo"), row.get("source_git_sha")))
    _require(len(png) == CASES and len(rgba) == CASES
             and families == {(CANDIDATES["pickr"]["repo"],
                               CANDIDATES["pickr"]["git_sha"])},
             "H190 previous private source authenticity or duplicate drift")
    return {"png": png, "rgba": rgba, "families": families}


def _check_ephemeral_source_capture(captures: list) -> list[dict]:
    _require(type(captures) is list and len(captures) == CASES*REPEATS,
             "H190 genuine original independent browser capture missing")
    output = []
    for spec in SPECS:
        rows = [entry for entry in captures if entry[0] == spec]
        _require(len(rows) == REPEATS,
                 "H190 source native task or replay missing")
        (first_spec, png, goal), (_, png2, goal2) = rows
        _require(type(png) is bytes and len(png) > 100
                 and png == png2 and goal == goal2,
                 "H190 original preaction screenshot/target replay drift")
        _require(type(goal) is dict and set(goal) == {"start", "drop"},
                 "H190 original reference drag geometry missing")
        for name in ("start", "drop"):
            p = goal[name]
            _require(type(p) is list and len(p) == 2
                     and all(type(x) is float and math.isfinite(x)
                             for x in p)
                     and 0 <= p[0] < WIDTH and 0 <= p[1] < HEIGHT,
                     "H190 invalid original source start/drop target point")
        with Image.open(BytesIO(png)) as source:
            _require(source.format == "PNG" and source.size == (WIDTH, HEIGHT),
                     "H190 original browser screenshot not 960x720 PNG")
            rgba_sha = sha256(source.convert("RGBA").tobytes()).hexdigest()
        output.append({
            "spec": spec, "png": png, "gold": goal,
            "image_sha256": sha256(png).hexdigest(),
            "visual_rgba_sha256": rgba_sha,
        })
    _require(len({x["image_sha256"] for x in output}) == CASES
             and len({x["visual_rgba_sha256"] for x in output}) == CASES,
             "H190 new private preaction source scene visual collision")
    return output


def quarantine_owner_dev_source(
    *, original_donor: Path, built_dist: Path, historical_root: Path,
    previous_train_private: Path, output_dir: Path, browser_channel: str,
) -> dict:
    """Native source proof and known-old/cross-split exclusions precede ALL disk writes."""
    destination = private_output_destination(output_dir)
    _require(isinstance(original_donor, Path) and original_donor.is_absolute()
             and isinstance(built_dist, Path) and built_dist.is_absolute()
             and isinstance(historical_root, Path) and historical_root.is_absolute(),
             "H190 original immutable source and history paths required")
    pin = verify_local_source_only(SOURCE_ID, original_donor)
    _require(pin["source_git_sha"] == CANDIDATES[SOURCE_ID]["git_sha"],
             "H190 immutable original interact.js git SHA changed")
    scratch_root = built_dist.parent
    for rel in (CANDIDATES[SOURCE_ID]["html_path"],
                CANDIDATES[SOURCE_ID]["license_path"],
                "examples/dropzones/index.js",
                "examples/dropzones/index.css"):
        donor_file = original_donor / rel
        scratch_file = scratch_root / rel
        _require(donor_file.is_file() and scratch_file.is_file()
                 and sha256(donor_file.read_bytes()).digest()
                 == sha256(scratch_file.read_bytes()).digest(),
                 "H190 derived original source scratch provenance mismatch")
    _require((built_dist / "dropzones/index.html").is_file(),
             "H190 source-only built original dropzones HTML missing")
    before_train = _read_previous_train_source(previous_train_private)
    _require(
        (CANDIDATES[SOURCE_ID]["repo"], CANDIDATES[SOURCE_ID]["git_sha"])
        not in before_train["families"],
        "H190 source-family train/dev overlap",
    )
    old = scan_legacy_manifest_sha_fields([historical_root])
    captures: list[tuple[dict, bytes, dict]] = []
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(Quiet, directory=str(built_dist.resolve(strict=True))),
    )
    Thread(target=srv.serve_forever, daemon=True).start()
    try:
        proof = run_native_preaction(
            srv.server_port, browser_channel, owner_old_sha_census=old,
            capture_preaction=lambda s, png, coordinates: captures.append(
                (dict(s), bytes(png), dict(coordinates))
            ),
        )
    finally:
        srv.shutdown()
        srv.server_close()
    _require(proof["original_preaction_scenes"] == CASES
             and proof["native_seed_and_target_successes"] == CASES*REPEATS*2
             and proof["dev_dataset_images_admitted"] == 0
             and proof["owner_old_sha_candidates_screened"] is True,
             "H190 actual original source replay or history collision check failed")
    scenes = _check_ephemeral_source_capture(captures)
    for row in scenes:
        _require(row["image_sha256"] not in before_train["png"]
                 and row["visual_rgba_sha256"] not in before_train["rgba"],
                 "H190 private TRAIN / proposed DEV images overlap")
    # No source media or owner reference target can be persisted before
    # old visual SHA exclusions, cross-source and native replay pass.
    destination.mkdir(mode=0o700, exist_ok=False)
    media = destination / "images"
    media.mkdir(mode=0o700)
    cases, gold = [], []
    for i, row in enumerate(scenes):
        spec = row["spec"]
        ident = f"h190-original-interact-{i:02d}"
        relative = f"images/{ident}.png"
        (destination / relative).write_bytes(row["png"])
        cases.append({
            "case_id": ident, "app": SOURCE_ID,
            "source_repo": CANDIDATES[SOURCE_ID]["repo"],
            "source_git_sha": CANDIDATES[SOURCE_ID]["git_sha"],
            "language": spec["language"],
            "instruction": spec["instruction"],
            "action_type": "drag", "image": relative,
            "image_sha256": row["image_sha256"],
            "visual_rgba_sha256": row["visual_rgba_sha256"],
            "purpose": "QUARANTINED_PROSPECTIVE_DEV_RESEARCH_ONLY",
            "train_eligible": False, "dev_eligible": False,
            "ood_eligible": False,
        })
        gold.append({
            "case_id": ident,
            "private_original_drag_start": row["gold"]["start"],
            "private_original_drop_target": row["gold"]["drop"],
            "not_for_model_input": True, "dev_eligible": False,
        })
    (destination / "source-prospects-not-admitted.json").write_text(
        json.dumps({
            "schema": SCHEMA, "status": "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION",
            "source_original_native_replayed": True,
            "known_old_sha_candidates_excluded": True,
            "different_original_source_from_train": True,
            "complete_prior_source_history_proven": False,
            "complete_prior_rgba_history_proven": False,
            "official_train_dev_dataset_admitted": False,
            "case_count": len(cases), "cases": cases,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (destination / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": gold}, indent=2) + "\n", encoding="utf-8",
    )
    return {
        "status": "H190_TWO_INTERACT_ORIGINAL_PREACTION_DEV_PROSPECTS_PRIVATE_QUARANTINED",
        "original_private_preaction_images": len(cases),
        "owner_private_native_drag_reference_records": len(gold),
        "disjoint_original_source_implementations_seen": 2,
        "known_old_png_hash_candidates_screened": len(old["_owner_only_png_sha_candidates"]),
        "dev_dataset_admitted": False, "model_calls": 0,
        "gpu_steps": 0, "old_ood_accessed": False, "public_images_uploaded": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H190 owner-only original interact.js DEV research capture, NOT DEV admission"
    )
    p.add_argument("--donor-root", type=Path, required=True)
    p.add_argument("--built-dist", type=Path, required=True)
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--previous-train-private", type=Path, required=True)
    p.add_argument("--owner-private-dir", type=Path, required=True)
    p.add_argument("--browser-channel", default="chromium")
    args = p.parse_args()
    out = quarantine_owner_dev_source(
        original_donor=args.donor_root, built_dist=args.built_dist,
        historical_root=args.historical_root,
        previous_train_private=args.previous_train_private,
        output_dir=args.owner_private_dir, browser_channel=args.browser_channel,
    )
    print(
        f"{out['status']} private_images={out['original_private_preaction_images']} "
        f"native_drag_goals={out['owner_private_native_drag_reference_records']} "
        f"distinct_original_apps={out['disjoint_original_source_implementations_seen']} "
        f"known_old_png_sha={out['known_old_png_hash_candidates_screened']} "
        "NO_TRAIN_DEV_ADMISSION_NO_GPU_NO_OLD_OOD_NO_PUBLIC_MEDIA"
    )


if __name__ == "__main__":
    main()

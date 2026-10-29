"""H194 two real original interact.js slider PRE-action DEV research PNGs.

PRIVATE owner-only quarantine, not admitted dataset. Copy-first H190/H191
quarantine and H193 native source+canvas replay. No old OOD/photos opened,
no paid GPU, no public image, labels, model or weights. Reference key
and original target center in separate owner-only non-model-input file.
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
import subprocess
from threading import Thread

from PIL import Image

from h174.source_candidates import CANDIDATES, verify_local_source_only
from h185.manifest_sha_only import (scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes)
from h188.owner_private_pickr_capture import _require
from h193.interact_snap_preaction import (
    REPEATS, SPECS, WIDTH, HEIGHT, run_original_snap,
)

SCHEMA = "hala.h194.owner-private-original-snap-slider-quarantine.v1"
STATUS = "H194_OWNER_PRIVATE_ORIGINAL_SLIDER_DEV_PROSPECTS_NOT_ADMITTED"
SOURCE = "interact-dropzones"
PREVIOUS = (
    ("Hala-H188-Private", "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION", 2, "pickr"),
    ("Hala-H190-Private", "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION", 2, SOURCE),
    ("Hala-H191-Private", "H191_TWELVE_PRIVATE_PICKR_PREACTION_QUARANTINE_ONLY", 12, "pickr"),
)


def _safe_lexical_path(path: Path) -> None:
    _require(isinstance(path, Path) and path.is_absolute(),
             "absolute owner-private source path required")
    for part in (path, *path.parents):
        _require(not part.is_symlink()
                 and not (hasattr(part, "is_junction") and part.is_junction())
                 and not (getattr(part.lstat(), "st_file_attributes", 0) & 0x400),
                 "owner-private symlink/junction/reparse path prohibited")


def _private_dest(path: Path) -> None:
    _require(isinstance(path, Path) and path.is_absolute()
             and path.name == "Hala-H194-Private"
             and not os.path.lexists(path),
             "new exact owner-private H194 output required; no overwrite")
    for parent in path.parents:
        _safe_lexical_path(parent)
        _require(parent.is_dir()
                 and not (parent / ".git").exists()
                 and not (parent / ".git").is_symlink(),
                 "owner-private output inside Git or unsafe ancestor")


def _prior_visual_sha_only(path: Path, status: str, count: int, family: str) -> dict:
    """Only new owner private provenance JSON; never prior image/gold."""
    _safe_lexical_path(path)
    _require(path.is_dir() and path.name in {p[0] for p in PREVIOUS},
             "missing prior source cohort")
    meta = path / "source-prospects-not-admitted.json"
    _safe_lexical_path(meta)
    _require(meta.is_file() and 0 < meta.stat().st_size < 256*1024,
             "prior private source SHA metadata missing/oversized")
    doc = json.loads(meta.read_text(encoding="utf-8"))
    _require(type(doc) is dict
             and doc.get("status") == status
             and doc.get("official_train_dev_dataset_admitted") is False
             and type(doc.get("cases")) is list
             and len(doc["cases"]) == count,
             "prior source cohort changed or became admitted")
    pin = CANDIDATES[family]
    png, rgba = set(), set()
    for row in doc["cases"]:
        _require(type(row) is dict
                 and row.get("source_repo") == pin["repo"]
                 and row.get("source_git_sha") == pin["git_sha"]
                 and all(row.get(key) is False
                         for key in ("train_eligible", "dev_eligible", "ood_eligible")),
                 "prior source family or eligibility drift")
        for key, group in (("image_sha256", png),
                           ("visual_rgba_sha256", rgba)):
            value = row.get(key)
            _require(type(value) is str and len(value) == 64
                     and all(ch in "0123456789abcdef" for ch in value),
                     "prior source visual fingerprint invalid")
            group.add(value)
    _require(len(png) == count and len(rgba) == count,
             "prior source images duplicate each other")
    return {"png": png, "rgba": rgba}


def _verify_original_built_snap_source(donor: Path, scratch: Path, dist: Path) -> None:
    _require(isinstance(donor, Path) and isinstance(scratch, Path)
             and isinstance(dist, Path)
             and all(p.is_absolute() for p in (donor, scratch, dist))
             and dist == scratch / ".h193-derived-original-snap",
             "built pinned source-only scratch path required")
    pin = verify_local_source_only(SOURCE, donor)
    _require(pin["source_git_sha"] == CANDIDATES[SOURCE]["git_sha"],
             "original donor Git SHA changed")
    for rel in ("examples/snap/index.html",
                "examples/snap/index.js",
                "examples/snap/index.css", "LICENSE"):
        origin_git_blob = subprocess.check_output(
            ["git", "-C", str(donor), "rev-parse", f"HEAD:{rel}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        original_scratch_file = scratch / rel
        _safe_lexical_path(original_scratch_file)
        _require(original_scratch_file.is_file(),
                 "original snapshot file missing in build")
        current_blob = subprocess.check_output(
            ["git", "hash-object", str(original_scratch_file)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        _require(origin_git_blob == current_blob,
                 "original source HTML/JS/CSS/LICENSE differs from pinned Git")
    _require((dist / "snap" / "index.html").is_file(),
             "built pinned original snap HTML absent")


def _replay_captures(captures: list) -> list[dict]:
    _require(type(captures) is list and len(captures) == len(SPECS)*REPEATS,
             "original independent native slider screenshots incomplete")
    result = []
    for spec in SPECS:
        rows = [x for x in captures if x[0] == spec]
        _require(len(rows) == REPEATS, "independent source scenario incomplete")
        (_, png, goal), (_, again_png, again_goal) = rows
        _require(type(png) is bytes and len(png) > 100
                 and png == again_png and goal == again_goal,
                 "original source screenshot/action geometry replay changed")
        _require(type(goal) is dict
                 and set(goal) == {"action_type", "key", "slider", "target_center"}
                 and goal["action_type"] == "key_press"
                 and goal["key"] == "ArrowRight"
                 and goal["slider"] == spec["target"],
                 "original model reference action malformed")
        pt = goal["target_center"]
        _require(type(pt) is list and len(pt) == 2
                 and all(type(v) is float and math.isfinite(v) for v in pt)
                 and 0 <= pt[0] < WIDTH and 0 <= pt[1] < HEIGHT,
                 "original target slider outside screenshot")
        with Image.open(BytesIO(png)) as im:
            _require(im.format == "PNG" and im.size == (WIDTH, HEIGHT),
                     "invalid original preaction source screenshot PNG")
            rgba = sha256(im.convert("RGBA").tobytes()).hexdigest()
        result.append({
            "spec": spec, "png": png, "goal": goal,
            "image_sha256": sha256(png).hexdigest(),
            "visual_rgba_sha256": rgba,
        })
    _require(len({r["image_sha256"] for r in result}) == len(SPECS)
             and len({r["visual_rgba_sha256"] for r in result}) == len(SPECS),
             "two original slider tasks have same preaction image pixels")
    return result


def quarantine_original_slider(
    *, original_donor: Path, original_scratch: Path, historical_root: Path,
    earlier_private_root: Path, private_output_dir: Path,
    browser_channel: str,
) -> dict:
    """Never write until actual source native rewards and all SHA checks pass."""
    _private_dest(private_output_dir)
    dist = original_scratch / ".h193-derived-original-snap"
    _verify_original_built_snap_source(original_donor, original_scratch, dist)
    _require(isinstance(historical_root, Path) and historical_root.is_absolute()
             and isinstance(earlier_private_root, Path)
             and earlier_private_root.is_absolute(),
             "owner historical and prior private roots required")
    old = scan_legacy_manifest_sha_fields([historical_root])
    previous = [
        _prior_visual_sha_only(
            earlier_private_root / name, status, expected, family
        )
        for name, status, expected, family in PREVIOUS
    ]

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(Quiet, directory=str(dist.resolve(strict=True))),
    )
    Thread(target=server.serve_forever, daemon=True).start()
    captures: list[tuple] = []
    try:
        native = run_original_snap(
            server.server_port, browser_channel, old,
            capture_preaction=lambda spec, png, gold: captures.append(
                (dict(spec), bytes(png), dict(gold))
            ),
        )
    finally:
        server.shutdown()
        server.server_close()
    _require(native["original_visual_source_tasks"] == 2
             and native["real_native_seed_and_target_keyboard_actions"] == 12
             and native["owner_old_sha_checked"] is True
             and native["train_or_dev_cases_admitted"] == 0
             and native["model_calls"] == 0,
             "real original native source/replay/historical SHA gate failed")
    verified = _replay_captures(captures)
    # Independent owner-private writer check, in addition to the H193
    # original browser replay receipt: a fabricated callback/metadata
    # must never bypass the conservative previously known old SHA gate.
    screen_unadmitted_new_case_hashes([
        {"image_sha256": row["image_sha256"],
         "visual_rgba_sha256": row["visual_rgba_sha256"]}
        for row in verified
    ], old)
    for row in verified:
        for prior in previous:
            _require(row["image_sha256"] not in prior["png"]
                     and row["visual_rgba_sha256"] not in prior["rgba"],
                     "H194 candidate repeats prior prospective TRAIN or DEV image")
    # All checks above are read-only: no previous image/gold opened.
    private_output_dir.mkdir(mode=0o700, exist_ok=False)
    (private_output_dir / "images").mkdir(mode=0o700)
    cases, gold = [], []
    for i, row in enumerate(verified):
        spec = row["spec"]
        case_id = f"h194-original-interact-snap-{i:02d}"
        relative = f"images/{case_id}.png"
        (private_output_dir / relative).write_bytes(row["png"])
        cases.append({
            "case_id": case_id,
            "app": "interact-snap",
            "source_repo": CANDIDATES[SOURCE]["repo"],
            "source_git_sha": CANDIDATES[SOURCE]["git_sha"],
            "language": spec["language"],
            "instruction": spec["instruction"],
            "target_slider": spec["target"],
            "action_type": "key_press",
            "image": relative,
            "image_sha256": row["image_sha256"],
            "visual_rgba_sha256": row["visual_rgba_sha256"],
            "purpose": "QUARANTINED_PROSPECTIVE_DEV_SLIDER_RESEARCH_ONLY",
            "train_eligible": False, "dev_eligible": False,
            "ood_eligible": False,
        })
        gold.append({
            "case_id": case_id,
            "private_original_action": row["goal"],
            "not_for_model_input": True,
            "dev_eligible": False,
        })
    (private_output_dir / "source-prospects-not-admitted.json").write_text(
        json.dumps({
            "schema": SCHEMA, "status": STATUS,
            "source_original_native_replayed": True,
            "all_known_old_shas_screened": True,
            "prior_train_dev_visual_disjoint": True,
            "historical_source_complete": False,
            "historical_decoded_rgba_complete": False,
            "official_train_dev_dataset_admitted": False,
            "case_count": len(cases),
            "cases": cases,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (private_output_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": gold}, indent=2) + "\n", encoding="utf-8",
    )
    return {
        "status": "H194_TWO_ORIGINAL_INTERACT_SNAP_PREACTION_IMAGES_QUARANTINED",
        "saved_private_pngs": len(cases),
        "separate_private_native_actions": len(gold),
        "existing_private_source_images_screened": sum(x[2] for x in PREVIOUS),
        "known_old_png_sha_candidates": len(old["_owner_only_png_sha_candidates"]),
        "official_train_dev_admitted": False,
        "model_calls": 0, "gpu_steps": 0,
        "old_ood_read": False, "public_images_uploaded": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="H194 genuine original snap slider source images, owner-private, nonadmitted"
    )
    p.add_argument("--donor-root", required=True, type=Path)
    p.add_argument("--scratch-root", required=True, type=Path)
    p.add_argument("--historical-root", required=True, type=Path)
    p.add_argument("--prior-private-root", required=True, type=Path)
    p.add_argument("--owner-private-dir", required=True, type=Path)
    p.add_argument("--browser-channel", default="msedge")
    args = p.parse_args()
    proof = quarantine_original_slider(
        original_donor=args.donor_root, original_scratch=args.scratch_root,
        historical_root=args.historical_root,
        earlier_private_root=args.prior_private_root,
        private_output_dir=args.owner_private_dir,
        browser_channel=args.browser_channel,
    )
    print(
        f"{proof['status']} private_pngs={proof['saved_private_pngs']} "
        f"private_native_actions={proof['separate_private_native_actions']} "
        f"prior_new_images_compared={proof['existing_private_source_images_screened']} "
        f"known_old_png_sha={proof['known_old_png_sha_candidates']} "
        "NO_DATA_ADMISSION_NO_GPU_NO_OLD_OOD_NO_PUBLIC_MEDIA"
    )


if __name__ == "__main__":
    main()

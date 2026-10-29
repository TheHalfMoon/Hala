"""Read ONLY new H197 PNGs; older cohorts are SHA metadata only.

Audits model-visible inputs independently of the original capture writer.
This does not promote any screenshot to TRAIN/DEV/OOD or touch old OOD.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)
from h194.owner_private_snap_capture import _safe_lexical_path
from h195.original_vanilla_datepicker_native import PIN, SCENES, WIDTH, HEIGHT
from h197.owner_calendar_private import NAME, STATUS, prior_shas, check


def audit(owner_dir: Path, old_private_root: Path,
          historical_root: Path) -> dict:
    check(owner_dir.is_absolute() and owner_dir.name == NAME,
          "new owner original calendar source path only")
    _safe_lexical_path(owner_dir)
    source = owner_dir / "source-prospects-not-admitted.json"
    ref = owner_dir / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
    for file in (source, ref):
        _safe_lexical_path(file)
        check(file.is_file() and 0 < file.stat().st_size < 256000,
              "missing or oversized private source/reference manifest")
    doc = json.loads(source.read_text(encoding="utf-8"))
    gold = json.loads(ref.read_text(encoding="utf-8"))
    check(set(doc) == {
        "status", "cases", "official_train_dev_dataset_admitted",
        "historical_source_complete", "historical_decoded_rgba_complete"
    } and doc["status"] == STATUS
          and doc["official_train_dev_dataset_admitted"] is False
          and doc["historical_source_complete"] is False
          and doc["historical_decoded_rgba_complete"] is False,
          "invalid private research-only source role")
    rows = doc["cases"]
    references = gold.get("rows")
    check(set(gold) == {"rows"} and type(rows) is list and
          type(references) is list and len(rows) == len(references) ==
          len(SCENES), "original source/reference separation mismatch")
    actual = []
    for i, (scene, row, reference) in enumerate(zip(SCENES, rows, references)):
        id = f"h197-calendar-{i:02d}"
        check(type(row) is dict and set(row) == {
            "case_id","source_repo","source_git_sha","language",
            "instruction","image","image_sha256","visual_rgba_sha256",
            "purpose","train_eligible","dev_eligible","ood_eligible"}
              and row["case_id"] == id
              and row["source_repo"] == PIN["repo"]
              and row["source_git_sha"] == PIN["git_sha"]
              and row["language"] == scene["language"]
              and row["instruction"] == scene["instruction"]
              and row["purpose"] == "RESEARCH_UNASSIGNED_NOT_ADMITTED"
              and all(row[k] is False for k in (
                  "train_eligible","dev_eligible","ood_eligible"))
              and row["image"] == f"images/{id}.png",
              "original pinned source identity or research-only split changed")
        image_path = owner_dir / row["image"]
        _safe_lexical_path(image_path)
        check(image_path.is_file() and 0 < image_path.stat().st_size < 8000000,
              "new original screenshot file missing or unsafe")
        blob = image_path.read_bytes()
        with Image.open(BytesIO(blob)) as image:
            check(image.format == "PNG" and image.size == (WIDTH,HEIGHT),
                  "new original datepicker screenshot dimensions invalid")
            rgba = sha256(image.convert("RGBA").tobytes()).hexdigest()
        encoded = sha256(blob).hexdigest()
        check(encoded == row["image_sha256"] and
              rgba == row["visual_rgba_sha256"],
              "owner new original screenshot PNG/RGBA SHA altered")
        check(type(reference) is dict and set(reference) == {
            "case_id","private_action","not_for_model_input"}
              and reference["case_id"] == id
              and reference["not_for_model_input"] is True,
              "private original date reference leaked into model input")
        goal = reference["private_action"]
        check(type(goal) is dict and set(goal) == {
            "action_type", "target_center", "native_selected_value"}
              and goal["action_type"] == "click"
              and goal["native_selected_value"] == scene["expected_value"],
              "original calendar reference reward drift")
        center = goal["target_center"]
        check(type(center) is list and len(center) == 2 and
              all(type(x) is float for x in center) and
              0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT,
              "original calendar reference outside source GUI")
        actual.append({
            "image_sha256": encoded, "visual_rgba_sha256": rgba,
        })
    check(len({x["image_sha256"] for x in actual}) == 2 and
          len({x["visual_rgba_sha256"] for x in actual}) == 2,
          "two new screenshot pixels duplicate")
    past_png, past_rgba = prior_shas(old_private_root)
    check(all(item["image_sha256"] not in past_png and
              item["visual_rgba_sha256"] not in past_rgba for item in actual),
          "new source images reused before in known private source cohort")
    history = scan_legacy_manifest_sha_fields([historical_root])
    screen_unadmitted_new_case_hashes(actual, history)
    return {
        "status": "H197_NEW_PRIVATE_CALENDAR_INDEPENDENT_RASTER_AUDIT_PASS",
        "new_png_rgba_rechecked": len(actual),
        "past_owner_image_shas_compared": len(past_png),
        "past_historical_png_shas_compared":
            len(history["_owner_only_png_sha_candidates"]),
        "train_dev_ood_admitted": False, "model_calls": 0,
        "old_ood_accessed": False,
    }


def main():
    p = argparse.ArgumentParser()
    for key in ("owner-private-dir","prior-private-root","historical-root"):
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    r = audit(args.owner_private_dir, args.prior_private_root,
              args.historical_root)
    print(f"{r['status']} "
          f"new={r['new_png_rgba_rechecked']} "
          f"prior_private={r['past_owner_image_shas_compared']} "
          f"old_png={r['past_historical_png_shas_compared']} "
          "NO_ADMISSION_NO_OLD_OOD_NO_MODEL_NO_UPLOAD")


if __name__ == "__main__":
    main()

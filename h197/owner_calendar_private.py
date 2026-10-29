"""Source-only owner research capture of two original calendar preactions.

Never an admitted training, development or OOD dataset. Previous private
cohorts contribute only source identity and SHA metadata, not images/gold.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)
from h194.owner_private_snap_capture import (
    _safe_lexical_path, _prior_visual_sha_only, PREVIOUS,
)
from h195.original_vanilla_datepicker_native import (
    PIN, SCENES, REPEATS, WIDTH, HEIGHT, run_original_datepicker,
)

NAME = "Hala-H197-Private"
STATUS = "H197_RESEARCH_CALENDAR_NOT_ADMITTED"
H194_STATUS = "H194_OWNER_PRIVATE_ORIGINAL_SLIDER_DEV_PROSPECTS_NOT_ADMITTED"


def check(ok, reason):
    if not ok:
        raise ValueError("H197: " + reason)


def prior_shas(root):
    a, b = set(), set()
    for name, status, count, family in PREVIOUS:
        item = _prior_visual_sha_only(root / name, status, count, family)
        check(not a.intersection(item["png"]) and
              not b.intersection(item["rgba"]), "prior cohort visual overlap")
        a.update(item["png"])
        b.update(item["rgba"])
    path = root / "Hala-H194-Private" / "source-prospects-not-admitted.json"
    _safe_lexical_path(path)
    check(path.is_file() and path.stat().st_size < 256000,
          "H194 source-only manifest missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    check(data.get("status") == H194_STATUS and
          data.get("official_train_dev_dataset_admitted") is False and
          len(data.get("cases", [])) == 2, "H194 source role changed")
    for row in data["cases"]:
        check(row["source_repo"] == "taye/interact.js" and
              all(row[x] is False for x in (
                  "train_eligible", "dev_eligible", "ood_eligible")),
              "H194 source identity or role changed")
        a.add(row["image_sha256"])
        b.add(row["visual_rgba_sha256"])
    check(len(a) == len(b) == 18, "existing private cohort hashes invalid")
    return a, b


def validate_captures(captures):
    check(type(captures) is list and len(captures) == 2*REPEATS,
          "missing original independent calendar captures")
    out = []
    for scene in SCENES:
        matching = [item for item in captures if item[0] == scene]
        check(len(matching) == REPEATS and matching[0] == matching[1],
              "original screenshot/target fresh-context drift")
        _, png, goal = matching[0]
        check(type(png) is bytes and len(png) > 100 and
              type(goal) is dict and
              set(goal) == {
                  "action_type", "target_center", "native_selected_value"
              } and goal["action_type"] == "click" and
              goal["native_selected_value"] == scene["expected_value"],
              "not a true original calendar native click")
        point = goal["target_center"]
        check(type(point) is list and len(point) == 2 and
              all(type(x) is float for x in point) and
              0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT,
              "native calendar click target outside screenshot")
        with Image.open(BytesIO(png)) as im:
            check(im.format == "PNG" and im.size == (WIDTH, HEIGHT),
                  "original screenshot pixels invalid")
            rgba = sha256(im.convert("RGBA").tobytes()).hexdigest()
        out.append({
            "scene": scene, "png": png, "goal": goal,
            "image_sha256": sha256(png).hexdigest(),
            "visual_rgba_sha256": rgba,
        })
    check(len({r["image_sha256"] for r in out}) == 2 and
          len({r["visual_rgba_sha256"] for r in out}) == 2,
          "duplicated original calendar screenshots")
    return out


def capture(donor, historical, older, output, channel):
    check(output.is_absolute() and output.name == NAME and
          not os.path.lexists(output), "new private output only; no overwrite")
    for parent in output.parents:
        _safe_lexical_path(parent)
        check(parent.is_dir() and not (parent / ".git").exists(),
              "private screenshots cannot be inside Git")
    old_png, old_rgba = prior_shas(older)
    history = scan_legacy_manifest_sha_fields([historical])
    captures = []
    proof = run_original_datepicker(
        donor, channel, historical,
        capture_preaction=lambda scene, png, goal: captures.append(
            (dict(scene), bytes(png), dict(goal))
        ),
    )
    check(proof["independent_original_date_selection_tasks"] == 2 and
          proof["old_git_checkouts_screened"] > 0 and
          proof["known_old_sha_screened"] is True and
          proof["train_dev_ood_admitted"] == 0,
          "real original source and old-history qualification failed")
    rows = validate_captures(captures)
    screen_unadmitted_new_case_hashes([
        {key: item[key] for key in ("image_sha256","visual_rgba_sha256")}
        for item in rows
    ], history)
    check(all(row["image_sha256"] not in old_png and
              row["visual_rgba_sha256"] not in old_rgba for row in rows),
          "new calendar reuses prior original image")
    check(not os.path.lexists(output), "output path created during screening")
    output.mkdir(mode=0o700)
    (output / "images").mkdir(mode=0o700)
    cases, references = [], []
    for index, row in enumerate(rows):
        scene = row["scene"]
        case_id = f"h197-calendar-{index:02d}"
        rel = f"images/{case_id}.png"
        (output / rel).write_bytes(row["png"])
        cases.append({
            "case_id": case_id, "source_repo": PIN["repo"],
            "source_git_sha": PIN["git_sha"],
            "language": scene["language"], "instruction": scene["instruction"],
            "image": rel, "image_sha256": row["image_sha256"],
            "visual_rgba_sha256": row["visual_rgba_sha256"],
            "purpose": "RESEARCH_UNASSIGNED_NOT_ADMITTED",
            "train_eligible": False, "dev_eligible": False,
            "ood_eligible": False,
        })
        references.append({
            "case_id": case_id, "private_action": row["goal"],
            "not_for_model_input": True,
        })
    (output / "source-prospects-not-admitted.json").write_text(
        json.dumps({
            "status": STATUS, "cases": cases,
            "official_train_dev_dataset_admitted": False,
            "historical_source_complete": False,
            "historical_decoded_rgba_complete": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").write_text(
        json.dumps({"rows": references}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(cases), len(old_png), len(history["_owner_only_png_sha_candidates"])


def main():
    p = argparse.ArgumentParser()
    for arg in ("donor-root","historical-root","prior-private-root","owner-private-dir"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--browser-channel", default="msedge")
    a = p.parse_args()
    n, old_private, historic = capture(
        a.donor_root, a.historical_root, a.prior_private_root,
        a.owner_private_dir, a.browser_channel,
    )
    print(f"H197_PRIVATE_ORIGINAL_CALENDAR_RESEARCH_PASS new={n} "
          f"prior_private_shas={old_private} old_historic_png_shas={historic} "
          "NO_ADMISSION_NO_OLD_OOD_NO_PUBLIC_MEDIA_NO_MODEL")


if __name__ == "__main__":
    main()

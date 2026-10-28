"""H173 gold-free unseen original-app OOD ZIP with mandatory replay receipt.

This is a source-only operation. It cannot read old OOD pixels, model
weights, or owner's native click coordinates. One-shot; never overwrite.
"""
import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from PIL import Image

from h173.capture_ood import (
    MANIFEST, CAPTURE_ATTEMPT_LOCK, REPLAY_ATTEMPT_LOCK,
    CAPTURE_LOCK_CONTENT, REPLAY_LOCK_CONTENT,
    validate_owner_source_and_reference,
)
from h173.source_preflight import OOD_COUNT, preregister_provisional_cases
from h173.stable_capture import original_visual_hash

SCHEMA = "hala.h173.private-unseen-original-two-app-ood-gold-free.v1"
REPLAY_SCHEMA = "hala.h173.original-ood-32-pixel-replay-receipt.v1"
PACK_ATTEMPT_LOCK = "H173_GOLD_FREE_PACK_ATTEMPT_CONSUMED.lock"
PACK_LOCK_CONTENT = "H173_GOLD_FREE_PACK_ATTEMPT_CONSUMED_NO_RETRY\n"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def pack(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.suffix != ".zip":
        raise ValueError("H173 one-shot private OOD zip must not preexist")
    # A JSON replay receipt alone is not evidence that the one-shot owner
    # capture and independent replay entry points were actually reserved.
    # Require their exact local attempt markers before packaging any image.
    for name, expected in (
        (CAPTURE_ATTEMPT_LOCK, CAPTURE_LOCK_CONTENT),
        (REPLAY_ATTEMPT_LOCK, REPLAY_LOCK_CONTENT),
    ):
        marker = source / name
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf8") != expected
        ):
            raise ValueError("H173 owner capture/replay attempt marker missing or altered")
    source_raw = (source / "manifest.json").read_bytes()
    doc = json.loads(source_raw)
    replay = json.loads(
        (source / "owner-source-replay-receipt.json").read_text("utf8")
    )
    if (
        replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != "H173_PRIVATE_NEW_OOD_32_EXACT_PIXEL_REPLAY_PASS"
        or replay.get("source_manifest_sha256") != sha(source_raw)
        or replay.get("cases") != OOD_COUNT
        or replay.get("independent_rgba_pixel_replays") != OOD_COUNT
        or replay.get("prior_train_dev_ood_excluded") is not True
        or replay.get("owner_reference_actions_not_in_model_input") is not True
        or replay.get("model_calls") != 0
        or replay.get("optimizer_steps") != 0
    ):
        raise ValueError("H173 missing genuine independent all-32 RGBA replay")
    gold_path = source / "owner-reference-private.json"
    if not gold_path.is_file():
        raise ValueError("H173 owner native reference evidence must be separate")
    owner_reference = json.loads(gold_path.read_text("utf8"))
    validate_owner_source_and_reference(doc, owner_reference)
    if (
        doc.get("schema") != MANIFEST
        or doc.get("split") != "OOD"
        or doc.get("count") != OOD_COUNT
        or doc.get("no_old_train_dev_ood_images") is not True
        or doc.get("no_model_calls_before_frozen_capture") is not True
        or doc.get("owner_gold_uploaded") is not False
    ):
        raise ValueError("H173 original OOD captured manifest not qualified")
    rows = doc.get("private_cases")
    plan = preregister_provisional_cases()
    if not isinstance(rows, list) or len(rows) != OOD_COUNT:
        raise ValueError("H173 must capture exactly 32 original scenes")
    fields = (
        "index", "split", "app", "family", "task", "action_type", "variant",
        "language", "image", "image_sha256", "instruction"
    )
    filtered, pictures, seen, visible = [], {}, set(), set()
    for expected, row in zip(plan, rows):
        if (
            any(row.get(k) != v for k, v in expected.items() if k != "split")
            or row.get("split") != "OOD"
            or row.get("original_reference_success") is not True
            or row.get("image") != f"images/{row['index']}.png"
            or row.get("image_sha256") in seen
            or not isinstance(row.get("instruction"), str)
            or len(row["instruction"]) < 8
            or any(k in row for k in (
                "owner_gold", "training_supervision", "reference_point",
                "owner_native_action", "owner_reference_actions",
            ))
        ):
            raise ValueError("H173 invalid/duplicated OOD case or leaked owner gold")
        raw = (source / row["image"]).read_bytes()
        if sha(raw) != row["image_sha256"]:
            raise ValueError("H173 original OOD screenshot SHA mismatch")
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.size != (960, 720):
                raise ValueError("H173 only 960x720 original PNGs admitted")
        rgba = original_visual_hash(raw)
        if rgba in visible:
            raise ValueError("H173 duplicate original GUI visual pixels with different PNG encoding")
        visible.add(rgba)
        seen.add(row["image_sha256"])
        pictures[row["image"]] = raw
        filtered.append({field: row[field] for field in fields})
    if (
        len(seen) != OOD_COUNT
        or sum(row["language"] == "ar" for row in filtered) != 16
        or sum(row["app"] == "tingle" for row in filtered) != 16
        or sum(row["app"] == "tiny-slider" for row in filtered) != 16
        or len(pictures) != OOD_COUNT
    ):
        raise ValueError("H173 source lacks 32 distinct bilingual two-app images")
    output_manifest = {
        "schema": SCHEMA,
        "split": "OOD",
        "count": OOD_COUNT,
        "never_previously_used_original_app_families": True,
        "previous_train_dev_ood_excluded": True,
        "owner_gold_uploaded": False,
        "owner_source_manifest_sha256": sha(source_raw),
        "independent_rgba_replay_receipt_sha256": sha(
            (source / "owner-source-replay-receipt.json").read_bytes()
        ),
        "private_cases": filtered,
    }
    # Reserve the source itself, not merely the requested output filename:
    # otherwise a caller could create multiple OOD ZIPs from the same 32
    # private images by changing --output. This is a fail-closed one-shot
    # attempt; a crash after reservation permanently consumes this source.
    pack_marker = source / PACK_ATTEMPT_LOCK
    try:
        with pack_marker.open("x", encoding="utf8") as guard:
            guard.write(PACK_LOCK_CONTENT)
            guard.flush()
            os.fsync(guard.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            "H173 gold-free OOD packaging already attempted; quarantine source"
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "manifest.json",
            json.dumps(
                output_manifest, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
        for path, png in sorted(pictures.items()):
            z.writestr(path, png)
    return {
        "status": "H173_PRIVATE_GOLD_FREE_ORIGINAL_OOD_ZIP_FROZEN_NOT_EVALUATED",
        "ood_cases": OOD_COUNT,
        "arabic": 16,
        "english": 16,
        "zip_sha256": sha(target.read_bytes()),
        "owner_gold_uploaded": False,
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.source, args.output), indent=2, sort_keys=True))

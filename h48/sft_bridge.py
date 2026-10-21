"""Create-only, source-verified H48 visual grounding SFT export.

Original browser PNGs remain in place; no model weights, GPU, or execution here.
Public DEV scenes are never included in the supervised training output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h48.audit import verify
from h48.factory import HEIGHT, PROVENANCE, SCHEMA, WIDTH, canonical

EXAMPLES = "h48-sft-authoring.jsonl"
MANIFEST = "h48-sft-manifest.json"


def source_rows(root: Path):
    """Reconstruct supervision from authentic source, pixels and DOM, not supplied labels."""
    root = Path(root)
    model, labels, evaluator, audit = verify(root)
    for filename, value in (
        ("model_inputs.json", model),
        ("authoring_labels.json", labels),
        ("evaluator_only.json", evaluator),
        ("audit.json", audit),
    ):
        if (root / filename).read_bytes() != canonical(value):
            raise ValueError("Existing H48 evidence projection changed: " + filename)
    oracles = {row["id"]: row for row in evaluator["cases"]}
    rows, media = [], []
    for label in labels["examples"]:
        original = oracles[label["id"]]
        if (label["split"] != "authoring" or original["split"] != "authoring"
                or label["source_rights"] != PROVENANCE):
            raise ValueError("Untrusted or nonauthoring H48 label")
        x = round(label["labelled_point"][0] * WIDTH)
        y = round(label["labelled_point"][1] * HEIGHT)
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            raise ValueError("Pixel point outside original screenshot")
        matched = []
        for item in original["observed_buttons"]:
            left, top, width, height = item["bounds"]
            if left <= x < left + width and top <= y < top + height:
                matched.append(item["index"])
        if matched != [original["expected_index"]]:
            raise ValueError("Supervised point is not the unique intended visible button")
        if label["labelled_target_code"] != original["expected_ref_code"]:
            raise ValueError("Grounding target label disagrees with source")
        if (label["image_sha256"] != original["image_sha256"]
                or label["width"] != WIDTH or label["height"] != HEIGHT):
            raise ValueError("Original screenshot identity changed")
        action = {"type": "click", "point": [x, y], "coordinate_space": "window",
                  "image_scale": 1, "element_ref_hint": "visual_target"}
        prompt = (
            "Use only the screenshot and goal to identify the intended visible control. "
            "Return one JSON click action with a window pixel point; do not act on the computer.\n"
            "Screenshot: 1024 by 640 pixels.\nGoal: " + label["goal"]
        )
        rows.append({
            "id": label["id"], "split": "train", "family": label["family"],
            "language": label["language"], "prompt": prompt,
            "completion": json.dumps(action, sort_keys=True, ensure_ascii=False),
            "image_path": label["image"], "image_sha256": label["image_sha256"],
        })
        media.append({
            "path": label["image"], "sha256": label["image_sha256"],
            "family": label["family"], "provenance": dict(PROVENANCE),
        })
    families = {row["family"] for row in rows}
    held_out = {row["family"] for row in evaluator["cases"] if row["split"] == "dev"}
    if (len(rows) != 64 or len({row["id"] for row in rows}) != 64
            or families & held_out or len(families) != 4 or len(held_out) != 2
            or sum(row["language"] == "ar" for row in rows) != 32
            or sum(row["language"] == "en" for row in rows) != 32):
        raise ValueError("H48 family- and language-isolation gate failed")
    return rows, media, audit, sorted(families), sorted(held_out)


def expected_bytes(root: Path):
    rows, media, audit, families, held_out = source_rows(root)
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    manifest = {
        "schema": "hala.training.manifest.v1",
        "split": "train",
        "source_kind": "synthetic",
        "source_schema": SCHEMA,
        "rights_cleared": True,
        "source_sha256": audit["source_sha256"],
        "capture_sha256": audit["capture_sha256"],
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "families": families,
        "held_out_families": held_out,
        "token_budget": 4096,
        "media": media,
        "label_scope": "H48 authentic browser screenshot: first unique visible target only",
        "arabic_image_training": {
            "enabled": True, "raqm_qualified": False,
            "render_route": "source_verified_chromium_edge",
            "capture_sha256": audit["capture_sha256"],
            "excluded_records": 0,
        },
        "training_executed": False,
        "adapter_qualified": False,
        "independent_blind_test": False,
    }
    return data, canonical(manifest), manifest


def export(root: Path):
    """Write two derived files only when absent; original evidence is immutable."""
    from training.artifacts import load_sft_artifact
    root = Path(root)
    examples, manifest_file = root / EXAMPLES, root / MANIFEST
    if examples.exists() or manifest_file.exists():
        raise FileExistsError("H48 SFT output is create-only")
    data, manifest_bytes, manifest = expected_bytes(root)
    created = []
    try:
        for path, payload in ((examples, data), (manifest_file, manifest_bytes)):
            with path.open("xb") as output:
                output.write(payload)
            created.append(path)
        load_sft_artifact(examples, manifest)
    except Exception:
        for path in created:
            path.unlink()
        raise
    return examples, manifest_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("h48/assets"))
    args = parser.parse_args()
    print(json.dumps({"output": [str(p) for p in export(args.root)],
                      "training_executed": False, "adapter_qualified": False}))

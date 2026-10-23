"""H68: source-bound readiness audit for a complete CPU-prepared H61 dataset.

A small 8-scene pilot may validate the transport; it must NOT be admitted as a
full H61 training corpus. The full gate demands four pinned original TRAIN
domains, 2,000 actual bilingual browser scenes, a 10k-row H61 curriculum and
its original source/provenance validator. A JSON receipt alone is not trusted.
Does not allocate GPUs, publish data, or execute model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from h60.source_fetch import REV
from h67.stage import validate_train

SCHEMA = "hala.h68.cpu-full-readiness.v1"
REQUIRED = {"slider": 3500, "slide_resize": 2500,
            "sheet": 2500, "text_highlight": 2500}


def _json(path, *, max_bytes: int = 2 * 1024 * 1024):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("source receipt missing, linked or oversized")
    result = json.loads(path.read_text("utf8"))
    if not isinstance(result, dict):
        raise ValueError("source receipt must be a JSON object")
    return result


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(work: Path, *, rederive: bool = True) -> dict:
    work = Path(work).resolve()
    errors = []
    domains = {}
    for domain, minimum in REQUIRED.items():
        directory = work / "export" / domain
        try:
            receipt = _json(directory / "receipt.json")
            jpeg_dir = directory / "images" / domain
            count = len(list(jpeg_dir.glob("*.jpg"))) if jpeg_dir.is_dir() else -1
            rows = directory / "labels.jsonl"
            if (receipt.get("schema") != "hala.h61.source-export.v1"
                    or receipt.get("source_domain") != domain
                    or receipt.get("source_commit_sha") != REV
                    or receipt.get("split") != "train"
                    or receipt.get("official_eval_or_test_used") is not False
                    or receipt.get("benchmark_answer_leakage") is not False
                    or receipt.get("images_wanted") != minimum
                    or receipt.get("original_images") != minimum
                    or receipt.get("native_drag_actions", 0) <= 0
                    or count != minimum or not rows.is_file()
                    or _sha(rows) != receipt.get("labels_sha256")):
                raise ValueError("original TRAIN source/byte/count qualification failed")
            domains[domain] = {"images": count,
                               "actions": receipt["native_drag_actions"]}
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{domain}: {type(error).__name__}: {str(error)[:120]}")

    fp_dir = work / "export" / "firstparty"
    fp_valid = False
    try:
        fp = _json(fp_dir / "receipt.json")
        labels = fp_dir / "labels.jsonl"
        actual_png = list((fp_dir / "png").glob("*.png"))
        if (fp.get("schema") != "hala.h61.firstparty-bilingual-drag.v1"
                or fp.get("scenes_authored") != 2000
                or fp.get("scenes_accepted") != 2000
                or fp.get("scenes_rejected") != 0
                or fp.get("by_language") != {"ar": 1000, "en": 1000}
                or len(actual_png) != 2000
                or _sha(labels) != fp.get("labels_sha256")):
            raise ValueError("incomplete 2000-scene bilingual browser-captured family")
        fp_valid = True
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"firstparty: {type(error).__name__}: {str(error)[:120]}")

    qualified = {}
    try:
        qualified = validate_train(work, rederive=rederive)
        # 10,000 SHA-backed media entries legitimately exceed 2 MiB.
        # Keep the strict 2 MiB receipt budget; only this derived TRAIN
        # manifest may use the existing 16 MiB H67 metadata budget.
        manifest = _json(work / "export" / "h61_drag_curriculum_manifest.json",
                         max_bytes=16 * 1024 * 1024)
        if (qualified["rows"] != 10000
                or qualified["media"] < 10000
                or manifest.get("target_rows") != 10000
                or manifest.get("family_targets", {}).get(
                    "h61_firstparty_bilingual_drag_train") != 2000
                or manifest.get("language_counts", {}).get("ar") != 1000
                or manifest.get("domains_without_whole_archive_verification") is None):
            raise ValueError("full H61 curriculum counts or family mix not qualified")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"curriculum: {type(error).__name__}: {str(error)[:120]}")

    return {
        "schema": SCHEMA,
        "status": "READY_FOR_PRIVATE_STAGE_BUILD" if not errors else "NOT_READY",
        "source_pinned_revision": REV,
        "domains": domains,
        "firstparty_complete": fp_valid,
        "curriculum_rows_verified": qualified.get("rows", 0),
        "errors": errors,
        "training_executed": False,
        "provider_byte_readback_verified": False,
        "t4_allocated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.work)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_PRIVATE_STAGE_BUILD" else 2


if __name__ == "__main__":
    raise SystemExit(main())

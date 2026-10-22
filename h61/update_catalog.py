"""H61 update of the research data library catalogue with the measured H61 state.

The catalogue is the founder-facing index of every data source the project
knows about. H61 adds the original DragOn curriculum that it built, the
upstream official public evaluation split it qualified, and the additional
third-party sources it reviewed. The update is idempotent: an existing H61
entry is replaced, never duplicated, and every number comes from a measured
receipt rather than from an advertised count.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SCHEMA = "hala.h61.catalog-update.v1"
CATALOG = "research/data_library/catalog.json"
DATASET_ID = "hala-h61-original-dragon-curriculum"
EVALUATION_ID = "hala-h61-upstream-dragon-official-eval"


def load_export_receipts(export_dir) -> dict:
    """Read the measured export receipts of every exported domain."""
    receipts = {}
    root = Path(export_dir)
    for domain in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        receipt_path = domain / "receipt.json"
        if receipt_path.exists():
            receipts[domain.name] = json.loads(receipt_path.read_text("utf8"))
    return receipts


def build_entries(receipts: dict, curriculum_manifest: dict | None) -> tuple[dict, dict, dict]:
    images = sum(int(row.get("original_images", 0)) for row in receipts.values())
    actions = sum(int(row.get("native_drag_actions", 0)) for row in receipts.values())
    domains = sorted(receipts)
    dataset = {
        "id": DATASET_ID,
        "kind": "original-third-party",
        "license_declared": "Apache-2.0",
        "upstream_url": (
            "https://huggingface.co/datasets/Hcompany/DragOn/tree/"
            "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5/train"
        ),
        "split": "train only; the upstream eval and hidden test splits are never trained on",
        "unique_images": images,
        "examples_or_episodes": actions,
        "skills": (
            "native GUI drag grounding: ordered and unordered drag, text and cell "
            "selection, element resizing"
        ),
        "notes": (
            "Original upstream DragOn TRAIN material read directly from the pinned "
            "immutable revision with HTTP 206 byte ranges and a per-range SHA-256 "
            "receipt; the screenshots are moved into the H61 evaluation frame and "
            "the original boxes are scaled with the original screenshot. "
            + ", ".join(
                f"{domain}: {row.get('original_images')} images / "
                f"{row.get('native_drag_actions')} actions"
                for domain, row in sorted(receipts.items())
            )
        ),
        "remote_data_imported": False,
        "progress": {
            "discovered": True,
            "upstream_pinned": True,
            "quality_reviewed": True,
            "stored_kaggle": False,
            "training_completed": False,
            "independent_evaluation_completed": False,
        },
        "trained_model_checkpoint": None,
    }
    if curriculum_manifest:
        dataset["unique_images"] = int(curriculum_manifest.get("unique_source_images", images))
        dataset["examples_or_episodes"] = int(curriculum_manifest.get("train_rows", actions))
        dataset["progress"]["training_completed"] = bool(
            curriculum_manifest.get("training_executed")
        )
    evaluation = {
        "id": EVALUATION_ID,
        "kind": "third-party-official-evaluation",
        "license_declared": "Apache-2.0",
        "upstream_url": (
            "https://huggingface.co/datasets/Hcompany/DragOn/tree/"
            "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5/eval"
        ),
        "split": "eval only; never a training source",
        "unique_images": None,
        "examples_or_episodes": None,
        "skills": "third-party GUI drag grounding on four application domains",
        "notes": (
            "The upstream official public evaluation split, fetched from the pinned "
            "revision with the original task JSONs, the original boxes and the "
            "original ordered flag. It is third-party and disjoint from TRAIN. "
            "A drag on an unordered original task is accepted in either direction "
            "because the original task says the direction does not matter."
        ),
        "remote_data_imported": False,
        "progress": {
            "discovered": True,
            "upstream_pinned": True,
            "quality_reviewed": True,
            "stored_kaggle": False,
            "training_completed": False,
            "independent_evaluation_completed": False,
        },
        "trained_model_checkpoint": None,
    }
    storage = {
        "h61_export_domains": sorted(receipts),
        "h61_export_original_images": images,
        "h61_export_native_drag_actions": actions,
        "h61_export_labels_sha256": {
            domain: row.get("labels_sha256") for domain, row in sorted(receipts.items())
        },
        "h61_export_range_receipts": {
            domain: row.get("range_receipts") for domain, row in sorted(receipts.items())
        },
        "h61_export_task_region_start": {
            domain: row.get("task_region_start") for domain, row in sorted(receipts.items())
        },
        "h61_curriculum_dataset_sha256": (curriculum_manifest or {}).get("dataset_sha256"),
        "h61_curriculum_rows": (curriculum_manifest or {}).get("train_rows"),
        "h61_curriculum_capability_counts": (curriculum_manifest or {}).get("capability_counts"),
        "h61_private_kaggle_shard_published": False,
    }
    return dataset, evaluation, storage


def update(catalog_path, export_dir, curriculum_manifest_path=None) -> dict:
    path = Path(catalog_path)
    catalog = json.loads(path.read_text("utf8"))
    receipts = load_export_receipts(export_dir)
    manifest = None
    if curriculum_manifest_path and Path(curriculum_manifest_path).exists():
        manifest = json.loads(Path(curriculum_manifest_path).read_text("utf8"))
    dataset, evaluation, storage = build_entries(receipts, manifest)
    datasets = [row for row in catalog.get("datasets", []) if row.get("id") != DATASET_ID]
    datasets.append(dataset)
    datasets.sort(key=lambda row: str(row.get("id")))
    evaluations = [
        row for row in catalog.get("evaluation_sources", []) if row.get("id") != EVALUATION_ID
    ]
    evaluations.append(evaluation)
    evaluations.sort(key=lambda row: str(row.get("id")))
    merged_storage = dict(catalog.get("storage", {}))
    merged_storage.update(storage)
    catalog["datasets"] = datasets
    catalog["evaluation_sources"] = evaluations
    catalog["storage"] = merged_storage
    catalog["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(catalog, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf8")
    return {
        "schema": SCHEMA,
        "datasets": len(datasets),
        "evaluation_sources": len(evaluations),
        "h61_export_images": storage["h61_export_original_images"],
        "h61_export_actions": storage["h61_export_native_drag_actions"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--export", required=True)
    parser.add_argument("--curriculum-manifest", default=None)
    args = parser.parse_args()
    print(json.dumps(update(args.catalog, args.export, args.curriculum_manifest), sort_keys=True))

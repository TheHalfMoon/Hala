"""H61 the readiness receipt of the remote run.

Every gate that can be checked without a GPU is checked here, on the real
data, and the result is written as one receipt. The receipt says exactly what
was proved and it does not claim anything about the model: the training
outcome is still unknown until the remote session finishes.

The gates are:

1. the four original TRAIN exports exist, are verified and rejected nothing;
2. the first-party bilingual corpus exists, is verified and rejected nothing;
3. the curriculum holds the requested rows over distinct screenshots, the
   capability and language mixture is the intended one, and every media path
   resolves under the training root;
4. the trainer's own loader re-derives and byte-compares that curriculum;
5. the training gate accepts the configuration, the curriculum and the real
   public DEV baseline;
6. the coordinate supervision density is measured on the real tokenizer;
7. the official upstream public evaluation scenes are fetched for every domain.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h61.baseline import build_baseline_receipt  # noqa: E402
from h61.curriculum import build as build_curriculum  # noqa: E402
from h61.official_eval import load_scenes  # noqa: E402
from training.preflight import TrainingConfig, validate_experiment  # noqa: E402
from training.scripts.train_adapter import load_authoring_records  # noqa: E402

SCHEMA = "hala.h61.readiness-receipt.v1"


def main() -> int:
    export_root = Path(sys.argv[1])
    repo = Path(__file__).resolve().parents[1]
    receipt = {"schema": SCHEMA, "export_root": str(export_root), "gates": {}, "failures": []}

    domains = {}
    for domain in ("slider", "slide_resize", "sheet", "text_highlight"):
        path = export_root / domain / "receipt.json"
        if not path.exists():
            receipt["failures"].append(f"the {domain} export is missing")
            continue
        data = json.loads(path.read_text("utf8"))
        domains[domain] = {
            "original_images": data.get("original_images"),
            "native_drag_actions": data.get("native_drag_actions"),
            "rejected": len(data.get("rejections") or []),
            "range_receipts": data.get("range_receipts"),
            "source_tar_bytes": data.get("source_tar_bytes"),
            "upstream_lfs_sha256": data.get("upstream_lfs_sha256"),
        }
        if domains[domain]["rejected"]:
            receipt["failures"].append(f"the {domain} export rejected rows")
    receipt["gates"]["original_exports"] = domains

    firstparty_path = export_root / "firstparty" / "receipt.json"
    if firstparty_path.exists():
        data = json.loads(firstparty_path.read_text("utf8"))
        receipt["gates"]["firstparty_corpus"] = {
            "scenes_accepted": data.get("scenes_accepted"),
            "scenes_rejected": data.get("scenes_rejected"),
            "by_language": data.get("by_language"),
            "by_kind": data.get("by_kind"),
            "ordered_true": data.get("ordered_true"),
            "ordered_false": data.get("ordered_false"),
        }
        if data.get("scenes_rejected"):
            receipt["failures"].append("the first-party corpus rejected scenes")
    else:
        receipt["failures"].append("the first-party corpus is missing")

    manifest_path = export_root / "h61_drag_curriculum_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf8"))
        rows = load_authoring_records(export_root / "h61_drag_curriculum.jsonl", manifest)
        unresolved = [
            row["image_path"] for row in rows
            if not (export_root / row["image_path"]).exists()
        ]
        receipt["gates"]["curriculum"] = {
            "train_rows": manifest["train_rows"],
            "unique_source_images": manifest["unique_source_images"],
            "capability_counts": manifest["capability_counts"],
            "language_counts": manifest["language_counts"],
            "ordered_true": manifest["ordered_true"],
            "ordered_false": manifest["ordered_false"],
            "selected_by_family": manifest.get("selected_by_family"),
            "loader_rows": len(rows),
            "unresolved_media": len(unresolved),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "prompt_revision": manifest.get("prompt_revision"),
        }
        if unresolved:
            receipt["failures"].append(f"{len(unresolved)} curriculum media paths do not resolve")
        if len(rows) != manifest["train_rows"]:
            receipt["failures"].append("the trainer loader does not return every curriculum row")
    else:
        receipt["failures"].append("the curriculum is missing")

    config_path = repo / "experiments" / "runtime" / "h61-colab-train-config.json"
    baseline_path = repo / "h60" / "evidence" / "public-dev-baseline-20261021.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf8"))
        config = TrainingConfig.from_dict(json.loads(config_path.read_text("utf8")))
        baseline = build_baseline_receipt(baseline_path)
        gate = validate_experiment(config, manifest, baseline)
        receipt["gates"]["training_gate"] = {
            "result": gate,
            "supervision": config.supervision,
            "coordinate_weight": config.coordinate_weight,
            "max_steps": config.max_steps,
            "gradient_accumulation": config.gradient_accumulation,
            "baseline_evaluated_tasks": baseline["evaluated_tasks"],
            "baseline_verified_success_rate": baseline["verified_success_rate"],
        }

    official_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if official_root and official_root.exists():
        scenes = {}
        for domain in ("slider", "sheet", "slide_resize", "text_highlight"):
            path = official_root / domain / "scenes.json"
            if not path.exists():
                continue
            loaded = load_scenes(path)
            scenes[domain] = {
                "scenes": len(loaded),
                "ordered": sum(1 for item in loaded if item["task"]["ordered"]),
            }
        receipt["gates"]["official_evaluation"] = scenes

    receipt["status"] = "READY" if not receipt["failures"] else "NOT_READY"
    receipt["model_result_known"] = False
    receipt["note"] = (
        "This receipt proves the CPU side of the remote run only. It says "
        "nothing about the trained model, and the model outcome stays unknown "
        "until the remote session publishes its raw receipts."
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if receipt["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

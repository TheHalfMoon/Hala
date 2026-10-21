"""H60 tests for the native drag SFT bridge and the public-dev diagnostic.

Both are pure CPU contracts: no model, no provider account, no upstream
source. The SFT bridge is exercised on a synthetic verified shard so the
original-source-to-training-label conversion is proved end to end.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

from PIL import Image


def _jpeg(colour: tuple[int, int, int], size: tuple[int, int] = (96, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def _build_shard(root: Path, domain: str = "slider", count: int = 6) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "images" / domain).mkdir(parents=True, exist_ok=True)
    (root / "tasks" / domain).mkdir(parents=True, exist_ok=True)
    rows = []
    members = []
    for index in range(count):
        key = str(uuid.UUID(int=index))
        payload = _jpeg((index * 7 % 255, 30, 120))
        (root / "images" / domain / f"{key}.jpg").write_bytes(payload)
        tasks = [
            {
                "image_id": key,
                "domain": domain,
                "ordered": index % 2 == 0,
                "intent": f"Original drag instruction {index}",
                "start_bbox": [10, 10, 30, 30],
                "end_bbox": [40, 20, 70, 50],
            }
        ]
        blob = json.dumps(tasks).encode("utf8")
        (root / "tasks" / domain / f"{key}.json").write_bytes(blob)
        members.append(
            {
                "image_id": key,
                "image_path": f"images/{domain}/{key}.jpg",
                "image_sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "image_bytes": len(payload),
                "image_width": 96,
                "image_height": 64,
            }
        )
        rows.append(
            {
                "id": f"h60-{domain}-{key}-t000",
                "split": "train",
                "domain": domain,
                "subtype": None,
                "language": "en",
                "image_id": key,
                "image_path": f"images/{domain}/{key}.jpg",
                "image_sha256": members[-1]["image_sha256"],
                "image_width": 96,
                "image_height": 64,
                "source_task_json_path": f"tasks/{domain}/{key}.json",
                "source_task_sha256": __import__("hashlib").sha256(blob).hexdigest(),
                "source_task_index": 0,
                "source_image_tar_offset": index * 1024,
                "source_task_tar_offset": index * 2048,
                "intent": f"Original drag instruction {index}",
                "action_type": "drag",
                "ordered": index % 2 == 0,
                "start_bbox": [10, 10, 30, 30],
                "end_bbox": [40, 20, 70, 50],
                "completion": {
                    "type": "drag",
                    "start_point": [20, 20],
                    "end_point": [55, 35],
                    "coordinate_space": "source_image_pixels",
                    "ordered": index % 2 == 0,
                },
            }
        )
    labels = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf8") + b"\n" for row in rows
    )
    (root / "labels.jsonl").write_bytes(labels)
    manifest = {
        "schema": "hala.h60.dragon-train-shard.v1",
        "shard_id": "h60-dragon-train-slider-0000",
        "source_repo": "Hcompany/DragOn",
        "source_commit_sha": "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
        "source_domain": domain,
        "split": "train",
        "source_license": "Apache-2.0",
        "upstream_full_tar_lfs_sha256": "0" * 64,
        "whole_tar_sha256_verified": True,
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
        "original_images": count,
        "original_task_members": count,
        "native_drag_actions": count,
        "ordered_true": sum(1 for row in rows if row["ordered"]),
        "ordered_false": sum(1 for row in rows if not row["ordered"]),
        "image_bytes": 0,
        "jpeg_structure_verified": count,
        "jpeg_pixels_decoded": 0,
        "labels_sha256": __import__("hashlib").sha256(labels).hexdigest(),
        "rejected": [],
        "source_window_ids": [],
        "original_image_members": members,
        "original_task_member_records": [],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )


class H60NativeDragSftTests(unittest.TestCase):
    def test_export_derives_rows_from_a_verified_shard(self) -> None:
        from h60 import native_drag_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_shard(root / "shard-0000")
            result = native_drag_sft.export(root)
            self.assertEqual(result["genuine_native_drag_rows"], 6)
            self.assertEqual(result["verified_original_screens"], 6)
            rows_path = root / native_drag_sft.ROWS_FILE
            manifest_path = root / native_drag_sft.MANIFEST_FILE
            self.assertTrue(rows_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text("utf8"))
            self.assertEqual(manifest["schema"], "hala.training.manifest.v1")
            self.assertEqual(manifest["split"], "train")
            self.assertTrue(manifest["rights_cleared"])
            self.assertEqual(manifest["source_kind"], "rights_cleared_public")
            self.assertEqual(manifest["token_budget"], 4096)
            self.assertTrue(manifest["no_benchmark_eval_source"])
            self.assertTrue(manifest["training_executed"] is False)
            self.assertTrue(manifest["model_improvement_qualified"] is False)
            self.assertEqual(len(manifest["families"]), 1)
            self.assertTrue(manifest["held_out_families"])
            self.assertFalse(set(manifest["families"]) & set(manifest["held_out_families"]))
            rows = [json.loads(line) for line in rows_path.read_text("utf8").splitlines()]
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(
                    set(row),
                    {"id", "split", "family", "language", "prompt", "completion",
                     "image_path", "image_sha256"},
                )
                completion = json.loads(row["completion"])
                self.assertEqual(
                    set(completion),
                    {"type", "start_point", "end_point", "coordinate_space", "ordered"},
                )
                self.assertEqual(completion["coordinate_space"], "source_image_pixels")
                self.assertIn(row["split"], ("train",))

    def test_export_is_create_only(self) -> None:
        from h60 import native_drag_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_shard(root / "shard-0000")
            native_drag_sft.export(root)
            with self.assertRaises(FileExistsError):
                native_drag_sft.export(root)

    def test_tampered_labels_are_rejected(self) -> None:
        from h60 import native_drag_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_shard(root / "shard-0000")
            native_drag_sft.export(root)
            rows_path = root / native_drag_sft.ROWS_FILE
            rows_path.write_text(
                rows_path.read_text("utf8").replace("drag", "click"), encoding="utf8"
            )
            manifest = json.loads((root / native_drag_sft.MANIFEST_FILE).read_text("utf8"))
            with self.assertRaises(ValueError):
                native_drag_sft.load_export(rows_path, manifest)

    def test_a_shard_without_a_verified_archive_is_rejected(self) -> None:
        from h60 import native_drag_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_shard(root / "shard-0000")
            manifest_path = root / "shard-0000" / "manifest.json"
            manifest = json.loads(manifest_path.read_text("utf8"))
            manifest["whole_tar_sha256_verified"] = False
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf8")
            with self.assertRaises(ValueError):
                native_drag_sft.derive(root)

    def test_the_baseline_gate_rejects_a_blind_test_claim(self) -> None:
        from h60 import native_drag_sft

        good = {
            "evidence": "real_inference",
            "task_kind": "mixed_ordered_unordered_drag",
            "held_out_source": "original_firstparty_public_dev_h55_h58_source_disjoint_not_training",
            "evaluated_tasks": 24,
            "held_out_dataset_sha256": "a" * 64,
            "independent_blind_test_completed": False,
        }
        native_drag_sft.validate_drag_baseline(good)
        blind = dict(good, independent_blind_test_completed=True)
        with self.assertRaises(ValueError):
            native_drag_sft.validate_drag_baseline(blind)
        wrong_kind = dict(good, task_kind="ordered_drag")
        with self.assertRaises(ValueError):
            native_drag_sft.validate_drag_baseline(wrong_kind)
        too_small = dict(good, evaluated_tasks=8)
        with self.assertRaises(ValueError):
            native_drag_sft.validate_drag_baseline(too_small)


class H60DiagnosticTests(unittest.TestCase):
    def _assets(self, root: Path) -> None:
        (root / "png").mkdir(parents=True, exist_ok=True)
        cases = []
        for index in range(2):
            ident = f"h55-en_case-{index:02d}"
            payload = _jpeg((10 * index, 20, 30), size=(1024, 640))
            (root / "png" / f"{ident}.png").write_bytes(payload)
            cases.append(
                {
                    "id": ident,
                    "language": "en",
                    "family": "fictional_firstparty_slider",
                    "start_point": [100 + index, 200],
                    "end_point": [300 + index, 200],
                }
            )
        (root / "capture.json").write_text(
            json.dumps({"cases": cases}, sort_keys=True), encoding="utf8"
        )
        (root / "source.json").write_text(json.dumps({"schema": "x"}), encoding="utf8")

    def test_format_failure_is_not_reported_as_grounding_failure(self) -> None:
        from h60 import diagnose_drag_outputs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "assets"
            self._assets(root)
            raw = Path(temporary) / "raw.json"
            cases = [
                {
                    "id": "h55-en_case-00",
                    "image_sha256": "0" * 64,
                    "raw_output": '```json\n{"type":"drag","start_point":[100,200],'
                                  '"end_point":[300,200],"coordinate_space":"source_image_pixels",'
                                  '"ordered":true}\n```',
                },
                {
                    "id": "h55-en_case-01",
                    "image_sha256": "0" * 64,
                    "raw_output": '```json\n{"type":"drag","start_point":[100,200],'
                                  '"end_point":[100,200],"coordinate_space":"source_image_pixels",'
                                  '"ordered":true}\n```',
                },
            ]
            raw.write_text(json.dumps({"cases": cases}), encoding="utf8")
            receipt = diagnose_drag_outputs.diagnose("h55", root, raw)
            self.assertEqual(receipt["totals"]["n"], 2)
            self.assertEqual(receipt["totals"]["strict_valid"], 0)
            self.assertEqual(receipt["totals"]["tolerant_valid"], 2)
            self.assertEqual(receipt["totals"]["geometrically_correct"], 1)
            self.assertEqual(receipt["totals"]["degenerate_start_equals_end"], 1)
            self.assertFalse(receipt["blind_test"])
            self.assertFalse(receipt["real_browser_model_task_success_verified"])


if __name__ == "__main__":
    unittest.main()

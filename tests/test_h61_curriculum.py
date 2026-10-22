"""H61 tests for the capability-balanced curriculum builder.

The curriculum is the part of H61 that decides what the model actually learns,
so these tests check the properties the founder asked for: capability balance,
screenshot deduplication, deterministic selection, a create-only export and a
tamper-detecting reload.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from h61.curriculum import (
    DOMAINS,
    PINNED_REVISION,
    build,
    capability_of,
    export,
    load_export,
    select,
)

TEMPLATE = {
    "id": "row",
    "split": "train",
    "domain": "slider",
    "subtype": None,
    "language": "en",
    "image_id": "image",
    "image_path": "images/slider/image.jpg",
    "image_sha256": "0" * 64,
    "image_width": 1024,
    "image_height": 640,
    "source_task_json_path": "tasks/slider/image.json",
    "source_task_sha256": "1" * 64,
    "source_task_index": 0,
    "source_image_tar_offset": 0,
    "source_task_tar_offset": 0,
    "intent": "Set the brightness to 6",
    "action_type": "drag",
    "ordered": True,
    "start_bbox": [591, 246, 601, 256],
    "end_bbox": [711, 246, 721, 256],
    "completion": {
        "type": "drag",
        "start_point": [596, 251],
        "end_point": [716, 251],
        "coordinate_space": "source_image_pixels",
        "ordered": True,
    },
}


def make_row(**overrides):
    row = json.loads(json.dumps(TEMPLATE))
    row.update(overrides)
    return row


def write_shard(root, domain, images, rows_per_image, ordered=True):
    shard = Path(root) / f"shard-{domain}"
    shard.mkdir(parents=True, exist_ok=True)
    labels = []
    for index in range(images):
        image_id = f"{domain}-{index:04d}"
        for task_index in range(rows_per_image):
            labels.append(
                make_row(
                    id=f"h60-{domain}-{image_id}-t{task_index:03d}",
                    domain=domain,
                    image_id=image_id,
                    image_path=f"images/{domain}/{image_id}.jpg",
                    ordered=ordered,
                    source_task_index=task_index,
                    end_bbox=[711 + task_index, 246, 721 + task_index, 256],
                    completion={
                        "type": "drag",
                        "start_point": [596, 251],
                        "end_point": [716 + task_index, 251],
                        "coordinate_space": "source_image_pixels",
                        "ordered": ordered,
                    },
                )
            )
    blob = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in labels
    ).encode("utf8")
    (shard / "labels.jsonl").write_bytes(blob)
    manifest = {
        "schema": "hala.h60.dragon-train-shard.v1",
        "shard_id": f"h60-dragon-train-{domain}-0000",
        "source_domain": domain,
        "split": "train",
        "source_commit_sha": PINNED_REVISION,
        "source_license": "Apache-2.0",
        "upstream_full_tar_lfs_sha256": "2" * 64,
        "whole_tar_sha256_verified": False,
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
        "original_images": images,
        "native_drag_actions": len(labels),
        "labels_sha256": hashlib.sha256(blob).hexdigest(),
    }
    (shard / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    return shard


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="h61-curriculum-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_capability_is_the_original_domain_and_direction(self):
        self.assertEqual(capability_of(make_row()), "slider/ordered")
        self.assertEqual(capability_of(make_row(ordered=False)), "slider/unordered")
        self.assertEqual(
            capability_of(make_row(subtype="cell-range")), "slider/ordered/cell-range"
        )

    def test_a_foreign_domain_is_refused(self):
        with self.assertRaises(ValueError):
            capability_of(make_row(domain="not_a_domain"))

    def test_selection_is_capability_balanced(self):
        rows = []
        for index in range(50):
            rows.append(make_row(id=f"a{index}", image_id=f"a{index}", domain="slider"))
        for index in range(50):
            rows.append(
                make_row(
                    id=f"b{index}",
                    image_id=f"b{index}",
                    domain="sheet",
                    ordered=False,
                    completion=dict(TEMPLATE["completion"], ordered=False),
                )
            )
        selected, balance = select(rows, 40, 1)
        self.assertEqual(len(selected), 40)
        self.assertEqual(balance["unique_screenshots"], 40)
        slider = [row for row in selected if row["domain"] == "slider"]
        sheet = [row for row in selected if row["domain"] == "sheet"]
        self.assertEqual(len(slider), 20)
        self.assertEqual(len(sheet), 20)

    def test_one_screenshot_cannot_saturate_the_mixture(self):
        rows = [make_row(id=f"r{index}", image_id="same", source_task_index=index)
                for index in range(20)]
        selected, balance = select(rows, 20, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(balance["unique_screenshots"], 1)

    def test_selection_is_deterministic(self):
        rows = [make_row(id=f"r{index}", image_id=f"i{index}") for index in range(30)]
        first, _ = select(rows, 20, 2)
        second, _ = select(list(reversed(rows)), 20, 2)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])

    def test_every_curriculum_media_path_resolves_under_the_training_root(self):
        """The trainer resolves a row's media against the training root.

        A path that is relative to the shard instead of the root is the
        defect that would stop a real training run on its first row, so the
        export proves the resolution here.
        """
        from PIL import Image

        shard = write_shard(self.root, "slider", 6, 2)
        images = shard / "images" / "slider"
        images.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            Image.new("RGB", (8, 8), (1, 2, 3)).save(images / f"slider-{index:04d}.jpg")
        export(self.root, target_rows=6, max_rows_per_image=1)
        rows = [
            json.loads(line)
            for line in (self.root / "h61_drag_curriculum.jsonl").read_text("utf8").splitlines()
            if line.strip()
        ]
        self.assertTrue(rows)
        for row in rows:
            resolved = self.root / row["image_path"]
            self.assertTrue(resolved.exists(), row["image_path"])
            self.assertTrue(str(resolved).startswith(str(self.root)))

    def test_export_is_create_only_and_reloads(self):
        write_shard(self.root, "slider", 30, 2)
        write_shard(self.root, "sheet", 30, 2, ordered=False)
        result = export(self.root, target_rows=40, max_rows_per_image=1)
        self.assertEqual(result["curriculum_rows"], 40)
        self.assertEqual(result["unique_source_images"], 40)
        manifest = json.loads(
            (self.root / "h61_drag_curriculum_manifest.json").read_text("utf8")
        )
        rows = load_export(self.root / "h61_drag_curriculum.jsonl", manifest)
        self.assertEqual(len(rows), 40)
        self.assertIn("slider/ordered", manifest["capability_counts"])
        self.assertIn("sheet/unordered", manifest["capability_counts"])
        with self.assertRaises(FileExistsError):
            export(self.root, target_rows=40, max_rows_per_image=1)

    def test_a_tampered_export_is_refused(self):
        write_shard(self.root, "slider", 10, 2)
        export(self.root, target_rows=10, max_rows_per_image=1)
        rows_path = self.root / "h61_drag_curriculum.jsonl"
        rows_path.write_bytes(rows_path.read_bytes() + b'{"tampered":true}\n')
        manifest = json.loads(
            (self.root / "h61_drag_curriculum_manifest.json").read_text("utf8")
        )
        with self.assertRaises(ValueError):
            load_export(rows_path, manifest)

    def test_a_shard_that_is_not_original_train_only_is_refused(self):
        shard = write_shard(self.root, "slider", 5, 1)
        manifest = json.loads((shard / "manifest.json").read_text("utf8"))
        manifest["benchmark_answer_leakage"] = True
        (shard / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
        with self.assertRaises(ValueError):
            build(self.root, target_rows=5, max_rows_per_image=1)

    def test_the_curriculum_records_the_capability_mix_it_actually_built(self):
        write_shard(self.root, "slider", 12, 3)
        blob, manifest, rows = build(self.root, target_rows=20, max_rows_per_image=2)
        self.assertEqual(manifest["train_rows"], len(rows))
        self.assertEqual(sum(manifest["capability_counts"].values()), len(rows))
        self.assertEqual(manifest["ordered_false"], 0)
        self.assertEqual(manifest["source_revision"], PINNED_REVISION)
        self.assertTrue(manifest["no_benchmark_eval_source"])
        self.assertFalse(manifest["model_improvement_qualified"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), manifest["dataset_sha256"])

    def test_every_declared_domain_is_part_of_the_original_set(self):
        self.assertEqual(set(DOMAINS), {"sheet", "slide_resize", "slider", "text_highlight"})


if __name__ == "__main__":
    unittest.main()

"""H48 supervised browser-grounding export: real-source positive and tamper tests."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from h48.sft_bridge import EXAMPLES, MANIFEST, export, expected_bytes, source_rows
from h48.audit import verify
from h48.factory import HEIGHT, WIDTH, SCHEMA
from h48.sft_bridge import load_export


ROOT = Path("h48/assets")


class H48SFTBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.manifest_bytes, cls.manifest = expected_bytes(ROOT)

    def test_original_training_split_no_dev_leakage_and_arabic(self):
        records = load_export(ROOT / EXAMPLES, self.manifest)
        self.assertEqual(len(records), 64)
        self.assertEqual(sum(row["language"] == "ar" for row in records), 32)
        self.assertEqual(sum(row["language"] == "en" for row in records), 32)
        self.assertEqual({r["family"] for r in records}, set(self.manifest["families"]))
        self.assertFalse(set(self.manifest["families"]) & set(self.manifest["held_out_families"]))
        self.assertEqual(self.manifest["source_schema"], SCHEMA)
        self.assertFalse(self.manifest["independent_blind_test"])
        self.assertFalse(self.manifest["training_executed"])
        self.assertFalse(self.manifest["adapter_qualified"])
        self.assertFalse(self.manifest["arabic_image_training"]["raqm_qualified"])
        self.assertEqual(self.manifest["arabic_image_training"]["render_route"],
                         "source_verified_chromium_edge")
        self.assertNotIn("observed_buttons", records[0]["prompt"])
        self.assertNotIn("labelled_point", records[0]["prompt"])
        self.assertNotIn("target_index", records[0]["prompt"])

    def test_immutable_export_and_unique_actual_dom_pixel_target(self):
        self.assertEqual((ROOT / EXAMPLES).read_bytes(), self.data)
        self.assertEqual((ROOT / MANIFEST).read_bytes(), self.manifest_bytes)
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), self.manifest["dataset_sha256"])
        _, labels, oracle, _ = verify(ROOT)
        by_id = {r["id"]: r for r in oracle["cases"]}
        points = [json.loads(json.loads(row)["completion"]) for row in self.data.decode().splitlines()]
        ids = [json.loads(row)["id"] for row in self.data.decode().splitlines()]
        for identity, action in zip(ids, points):
            matched = []
            x, y = action["point"]
            self.assertLess(x, WIDTH)
            self.assertLess(y, HEIGHT)
            for button in by_id[identity]["observed_buttons"]:
                a, b, w, h = button["bounds"]
                if a <= x < a + w and b <= y < b + h:
                    matched.append(button["index"])
            self.assertEqual(matched, [by_id[identity]["expected_index"]])

    def test_h14_immutable_training_loader_source_is_not_changed(self):
        freeze = json.loads(Path("evaluation/heldout/manifest.json").read_text("utf-8"))
        original = Path("training/artifacts.py").read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         freeze["files"]["training/artifacts.py"])

    def test_create_only_rejects_overwrite(self):
        with self.assertRaises(FileExistsError):
            export(ROOT)

    def test_manifest_forgery_rights_render_and_split_rejected(self):
        modifiers = (
            lambda m: m.update(source_sha256="0" * 64),
            lambda m: m["media"][0]["provenance"].update(license="unknown"),
            lambda m: m["arabic_image_training"].update(raqm_qualified=True),
            lambda m: m["arabic_image_training"].update(render_route="unverified"),
            lambda m: m.update(held_out_families=m["families"][:]),
            lambda m: m.update(source_schema="unknown"),
            lambda m: m.update(independent_blind_test=True),
        )
        for i, change in enumerate(modifiers):
            with self.subTest(modification=i):
                manifest = copy.deepcopy(self.manifest)
                change(manifest)
                with self.assertRaises((ValueError, KeyError)):
                    load_export(ROOT / EXAMPLES, manifest)

    def test_modified_training_label_fails_even_with_rehashed_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "h48-original"
            shutil.copytree(ROOT, root)
            lines = (root / EXAMPLES).read_text("utf-8").splitlines()
            first = json.loads(lines[0])
            action = json.loads(first["completion"])
            action["point"] = [0, 0]
            first["completion"] = json.dumps(action, sort_keys=True)
            lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
            (root / EXAMPLES).write_text("\n".join(lines) + "\n", encoding="utf-8")
            forged = copy.deepcopy(self.manifest)
            forged["dataset_sha256"] = hashlib.sha256((root / EXAMPLES).read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_export(root / EXAMPLES, forged)


if __name__ == "__main__":
    unittest.main()

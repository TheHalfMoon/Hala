"""H357 original-source GUI TRAIN/DEV candidate engineering, no admission.

All ephemeral image/gold samples are generated under TemporaryDirectory and
deleted. No private OOD, models, GPU or source rights are claimed.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import parse_actions
from training.preflight import require_admitted_gui_training_manifest
from h357.source_candidate import (
    SCHEMA, ROLE, COUNT, SOURCE, candidate_specs, capture_source_candidate,
)


class H357PublicOriginalSourceCandidateTests(unittest.TestCase):
    def test_new_original_single_source_has_unadmitted_candidate_only(self):
        specs = candidate_specs()
        self.assertEqual(len(specs), 64)
        self.assertEqual(Counter(r["language"] for r in specs),
                         {"en": 32, "ar": 32})
        self.assertEqual({r["family"] for r in specs},
                         {"controlgarden-single-app"})
        self.assertEqual({r["split"] for r in specs},
                         {"TRAIN_DEV_CANDIDATE_NOT_ADMITTED"})
        self.assertEqual(len({r["case_id"] for r in specs}), COUNT)
        self.assertEqual(specs, candidate_specs())  # deterministic
        for row in specs:
            self.assertEqual(len(row["source_tiles"]), 4)
            self.assertEqual(len({shape for shape, _ in row["source_tiles"]}), 4)
            self.assertEqual(len({color for _, color in row["source_tiles"]}), 4)
            self.assertEqual(len(set(tuple(x) for x in row["source_tiles"])), 4)
            self.assertIn("اختر" if row["language"] == "ar" else "Select",
                          row["task"])

    def test_zero_official_training_rights_or_admission_even_self_declared(self):
        with self.assertRaisesRegex(ValueError, "NOT_ADMITTED"):
            require_admitted_gui_training_manifest(
                b'{"schema":"hala.training.manifest.v1","rights_cleared":true}')

        self.assertTrue(SOURCE.exists())
        self.assertIn("TRAIN_DEV_REVIEW_CANDIDATE_ONLY", ROLE)

    @unittest.skipUnless(os.environ.get("HALA_H357_BROWSER_QA") == "1" and importlib.util.find_spec("playwright") is not None,
                         "optional Playwright; dedicated H357 source CI installs it")
    def test_actual_original_browser_64_replayed_gold_correct_private_temp(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "ephemeral-original-candidates"
            report = capture_source_candidate(
                root, browser_kind="msedge" if os.name == "nt" else "chromium")
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["role"], ROLE)
            self.assertEqual(report["original_native_browser_successes"], COUNT)
            self.assertEqual(report["exact_pixel_double_captures"], COUNT)
            self.assertEqual(report["language_counts"], {"en": 32, "ar": 32})
            self.assertFalse(report["official_train_admitted"])
            self.assertFalse(report["independent_rights_reviewed"])
            self.assertFalse(report["fresh_blind_ood"])
            self.assertFalse(report["can_train"])
            self.assertEqual(report["gpu_calls"], 0)
            rows = [json.loads(line) for line in
                    (root / "candidate-rows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), COUNT)
            self.assertEqual(len({r["image_sha256"] for r in rows}), COUNT)
            for record in rows:
                self.assertIn("screenshot-only", record["prompt"])
                action = parse_actions(record["completion"], training_target=True)
                self.assertEqual(action["actions"][0]["kind"], "click")
                self.assertEqual(record["split"], "TRAIN_DEV_CANDIDATE_NOT_ADMITTED")
                self.assertEqual(record["family"], "controlgarden-single-app")
                image = root / record["image_path"]
                self.assertTrue(image.is_file())
                self.assertTrue(image.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a")))
            # No external private data, no model artifacts, no OOD transcript.
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"images", "candidate-rows.jsonl", "candidate-manifest.json"})


if __name__ == "__main__":
    unittest.main()

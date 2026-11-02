"""H358 source-only English/Arabic multistep browser workflow qualification.

No private OOD, actual model inference, training rights, paid compute or
weights. Generated screenshots and labels never leave a TemporaryDirectory.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h199.native_action_sft import (
    parse_actions, ADMITTED_TRAIN_MANIFEST_SHAS,
)
from h355.signed_rights_intake import TRUSTED_REVIEWERS
from h356.sealed_ood_custody import TRUSTED_CUSTODIANS
from training.preflight import require_admitted_gui_training_manifest
from h358.source_candidate import (
    SOURCE, CASES, COUNT, ROLE, SCHEMA, FAMILY, STEPS,
    scenes, capture_candidate,
)


class H358OriginalMultistepCandidateTests(unittest.TestCase):
    def test_32_unique_synthetic_source_scenes_and_action_phases(self):
        roster = scenes()
        self.assertEqual(len(roster), 32)
        self.assertEqual(Counter(row["language"] for row in roster),
                         {"en": 16, "ar": 16})
        self.assertEqual(Counter(row["theme"] for row in roster),
                         {"light": 16, "dark": 16})
        self.assertEqual({row["family"] for row in roster}, {FAMILY})
        self.assertEqual({row["split"] for row in roster}, {ROLE})
        # H362: each bilingual pair shares identical nonlinguistic
        # workspace, synthetic value and theme; language is the only
        # changed task/UI dimension, preventing prefix/number leakage.
        self.assertEqual(len({row["code"] for row in roster}), CASES // 2)
        for index in range(0, CASES, 2):
            english, arabic = roster[index], roster[index + 1]
            self.assertEqual((english["code"], english["workspace"],
                              english["theme"]),
                             (arabic["code"], arabic["workspace"],
                              arabic["theme"]))
            self.assertEqual((english["language"], arabic["language"]),
                             ("en", "ar"))
        self.assertEqual(roster, scenes())
        self.assertEqual(STEPS, ("open", "focus", "type", "save"))
        self.assertEqual(COUNT, CASES * 4)
        for row in roster:
            self.assertIn(row["code"], row["task"])
            self.assertIn(("أنشئ" if row["language"] == "ar" else "Create"),
                          row["task"])

    def test_no_self_approval_or_unseen_ood_even_with_new_original_source(self):
        self.assertEqual(ADMITTED_TRAIN_MANIFEST_SHAS, frozenset())
        self.assertEqual(len(TRUSTED_REVIEWERS), 0)
        self.assertEqual(len(TRUSTED_CUSTODIANS), 0)
        with self.assertRaisesRegex(ValueError, "NOT_ADMITTED"):
            require_admitted_gui_training_manifest(
                b'{"rights_cleared":true, "source":"noterelay"}')
        raw = SOURCE.read_bytes()
        self.assertLess(len(raw), 25000)
        self.assertNotIn(b"http://", raw)
        self.assertNotIn(b"https://", raw)
        self.assertNotIn(b"<script src=", raw)
        self.assertNotIn(b"fetch(", raw)
        self.assertIn(b"window.getOriginalState", raw)

    @unittest.skipUnless(
        os.environ.get("HALA_H358_BROWSER_QA") == "1" and
        importlib.util.find_spec("playwright") is not None,
        "requires explicit browser-QA opt-in; pinned Playwright on H358 CI")
    def test_real_browser_32_original_workflows_128_pre_action_screen_labels(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "ephemeral-original-multistep"
            manifest = capture_candidate(
                out, browser_kind="msedge" if os.name == "nt" else "chromium")
            self.assertEqual(manifest["schema"], SCHEMA)
            self.assertEqual(manifest["role"], ROLE)
            self.assertEqual(manifest["case_count"], CASES)
            self.assertEqual(manifest["rows"], COUNT)
            self.assertEqual(manifest["original_native_state_successes"], COUNT)
            self.assertEqual(manifest["stable_initial_screenshots"], CASES)
            self.assertEqual(manifest["language_cases"], {"en": 16, "ar": 16})
            for field in ("independent_rights_approved", "official_train_admitted",
                          "fresh_blind_ood", "has_source_disjoint_dev",
                          "can_train", "can_release"):
                self.assertIs(manifest[field], False)
            self.assertEqual(manifest["actual_model_calls"], 0)
            self.assertEqual(manifest["gpu_calls"], 0)
            raw = (out / "candidate-rows.jsonl").read_text(encoding="utf-8")
            rows = [json.loads(row) for row in raw.splitlines()]
            self.assertEqual(len(rows), COUNT)
            self.assertEqual(len({r["image_sha256"] for r in rows}), COUNT)
            self.assertEqual(len({r["id"] for r in rows}), COUNT)
            self.assertEqual(
                Counter(r["phase"] for r in rows),
                {phase: CASES for phase in STEPS})
            self.assertEqual(
                Counter(r["language"] for r in rows), {"en": 64, "ar": 64})
            for row in rows:
                with self.subTest(row=row["id"]):
                    self.assertEqual(row["family"], FAMILY)
                    self.assertEqual(row["split"], ROLE)
                    action = parse_actions(row["completion"], training_target=True)
                    expected = "type_text" if row["phase"] == "type" else "click"
                    self.assertEqual(action["actions"][0]["kind"], expected)
                    self.assertIn("screenshot-only", row["prompt"])
                    image = out / row["image_path"]
                    self.assertTrue(image.is_file())
                    self.assertTrue(image.read_bytes().startswith(
                        bytes.fromhex("89504e470d0a1a0a")))
            self.assertEqual(
                {path.name for path in out.iterdir()},
                {"images", "candidate-rows.jsonl", "candidate-manifest.json"})


if __name__ == "__main__":
    unittest.main()

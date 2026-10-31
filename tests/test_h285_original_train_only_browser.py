"""H285 FIRST-PARTY TRAIN-CANDIDATE ONLY; browser fixtures are not model scores.

Real hosted Chromium/Edge checks use fixed ORIGINAL source-owned
supervision. They never train Qwen, open blind DEV/OOD, or persist PNGs.
"""
from __future__ import annotations

from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, HEIGHT, WIDTH, parse_actions,
)
from h285.train_only_browser_candidates import (
    IDS, ROLE, SCENES, SOURCE_BLOB, checked_source,
    exact_task, produce_train_only_batch,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "h285" / "shiftdesk.html"


class H285TrainOnlySourceContractTests(unittest.TestCase):
    def test_authored_original_bytes_and_24_registered_scenes(self):
        path, url = checked_source(ORIGINAL)
        self.assertEqual(path, ORIGINAL)
        self.assertTrue(url.startswith("file:"))
        self.assertEqual(SOURCE_BLOB,
                         "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac")
        self.assertEqual(len(IDS), 8)
        self.assertEqual(len(SCENES), 24)
        self.assertEqual(len(set(SCENES)), 24)
        self.assertEqual(sum(s[1] == "en" for s in SCENES), 12)
        self.assertEqual(sum(s[1] == "ar" for s in SCENES), 12)
        self.assertEqual(len(set(s[0] for s in SCENES)), 24)

    def test_no_dev_ood_or_training_admission_promotion(self):
        self.assertEqual(ROLE, "TRAIN_CANDIDATE_ONLY_NOT_OFFICIALLY_ADMITTED")
        self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)
        for scene in SCENES:
            task = exact_task(scene)
            self.assertIn(scene[-1], task)
            self.assertNotIn("OOD", task)
        with self.assertRaises(ValueError):
            exact_task(("fake-split", "en", 0, "light", "standard", IDS[0]))

    def test_tampered_source_bytes_refused_without_browser(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "h285"
            target.mkdir()
            candidate = target / "shiftdesk.html"
            candidate.write_bytes(ORIGINAL.read_bytes() + b"<!-- tamper -->")
            with self.assertRaisesRegex(ValueError, "byte history drift"):
                checked_source(candidate)
        with self.assertRaises(ValueError):
            checked_source(Path("shiftdesk.html"))

    def test_wrong_role_and_noncallback_refused_before_opening_browser(self):
        for bad_role in ("TRAIN", "DEV", "OOD", "BLIND", "TRAIN_ADMITTED", None):
            with self.subTest(bad_role=bad_role), self.assertRaises(ValueError):
                produce_train_only_batch(
                    None, ORIGINAL, lambda png, prompt, action: None,
                    role=bad_role)
        with self.assertRaisesRegex(ValueError, "sink required"):
            produce_train_only_batch(None, ORIGINAL, None, role=ROLE)

    def test_duplicate_frame_blocks_all_candidate_training_sink_writes(self):
        """No partial TRAIN candidate emission after any duplicate frame."""
        emitted = []
        row = dict(scene=SCENES[0][0], lang="en", png=b"x",
                   png_sha256="0" * 64, rgba_sha256="1" * 64,
                   user_task="invented", completion="invented",
                   source_state_success=True)
        calls = 0
        def duplicate(*args):
            nonlocal calls
            calls += 1
            return dict(row, scene=SCENES[(calls-1) % 24][0])
        with patch("h285.train_only_browser_candidates._candidate_one",
                   side_effect=duplicate):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                produce_train_only_batch(
                    object(), ORIGINAL,
                    lambda *args: emitted.append(args), role=ROLE)
        self.assertEqual(calls, 2)
        self.assertEqual(emitted, [])


@unittest.skipUnless(os.environ.get("HALA_H285_BROWSER_QUALIFY") == "1",
                     "real browser TRAIN-candidate source tests require opt-in")
class H285OriginalTrainOnlyRealBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H285_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("H285 browser must be Chromium or Edge")
            options = ({"headless": True, "channel": "msedge"}
                       if channel == "msedge"
                       else {"headless": True, "args": ["--no-sandbox"]})
            cls.browser = cls.pw.chromium.launch(**options)
        except BaseException:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_all_24_actual_browser_source_supervisions(self):
        seen = set()
        counter = {"count": 0, "en": 0, "ar": 0}

        def sink(png, prompt, label):
            # Sample stays ONLY in process memory and never enters logs.
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertLess(len(png), 4 * 1024 * 1024)
            self.assertIn("Task: ", prompt)
            self.assertEqual(
                parse_actions(label, training_target=True)["actions"][0]["kind"],
                "click")
            key = sha256(png).hexdigest()
            self.assertNotIn(key, seen)
            seen.add(key)
            counter["count"] += 1
            for language, indicator in (("en", "Complete fictional"),
                                        ("ar", "أنجز بطاقة")):
                if indicator in prompt:
                    counter[language] += 1
                    break

        report = produce_train_only_batch(
            self.browser, ORIGINAL, sink, role=ROLE)
        self.assertEqual(counter, {"count": 24, "en": 12, "ar": 12})
        self.assertEqual(len(seen), 24)
        self.assertEqual(report["actual_browser_source_actions_successful"], 24)
        self.assertEqual(report["distinct_png_screenshots"], 24)
        self.assertEqual(report["distinct_decoded_rgba_screenshots"], 24)
        self.assertEqual(report["english_candidates"], 12)
        self.assertEqual(report["arabic_candidates"], 12)
        self.assertEqual(report["valid_h199_supervision_labels"], 24)
        self.assertFalse(report["official_train_manifest_admitted"])
        self.assertFalse(report["independent_dev_ood_evaluated"])
        self.assertFalse(report["adapter_model_improvement_proven"])
        self.assertFalse(report["public_private_media_or_labels_saved"])
        self.assertEqual(report["pretrained_model_generations"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertNotIn("png", json.dumps(report).lower().replace("distinct_png_screenshots", ""))
        self.assertNotIn("completion", report)

    def test_labeling_never_admits_source_as_blind_ood(self):
        report = produce_train_only_batch(
            self.browser, ORIGINAL, lambda *_: None, role=ROLE)
        self.assertTrue(report["role"].startswith("TRAIN_CANDIDATE"))
        self.assertFalse(report["independent_dev_ood_evaluated"])
        self.assertFalse(report["old_consumed_ood_opened"])


if __name__ == "__main__":
    unittest.main()

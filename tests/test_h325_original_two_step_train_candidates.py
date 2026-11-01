"""H325: invented-only unit tests plus opt-in original Chromium/Edge actions."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from h325.two_step_train_candidates import (
    SOURCE_BLOB, ROLE, SCENES, IDS, checked_source, exact_task,
    produce_two_step_train_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h325" / "railkit.html"


class H325CandidateContract(unittest.TestCase):
    def test_source_pin_and_original_scene_design(self):
        self.assertEqual(checked_source(SOURCE)[0], SOURCE)
        self.assertEqual(SOURCE_BLOB, "743b5e1b54774a1deea4dbe6bfbe0613adead7e4")
        self.assertEqual(len(SCENES), 16)
        self.assertEqual(len(set(SCENES)), 16)
        self.assertEqual(sum(c[1] == "en" for c in SCENES), 8)
        self.assertEqual(sum(c[1] == "ar" for c in SCENES), 8)
        self.assertEqual({c[2] for c in SCENES}, {0,1,2,3})
        self.assertEqual({c[3] for c in SCENES}, {False, True})
        self.assertEqual({c[4] for c in SCENES}, {0,1,2,3})
        for scene in SCENES:
            self.assertIn(IDS[scene[4]], exact_task(scene))
            self.assertIn(scene[5], exact_task(scene))
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("h325/railkit.html -text", attrs)

    def test_tampered_source_is_not_accepted(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "h325"
            target.mkdir()
            changed = target / "railkit.html"
            changed.write_bytes(SOURCE.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                checked_source(changed)

    def test_unregistered_scene_and_roles_denied_before_browser(self):
        calls = []
        for role in ("TRAIN", "ADMITTED_TRAIN", "DEV", "BLIND_OOD", "RELEASE"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                produce_two_step_train_candidates(
                    None, SOURCE, lambda *args: calls.append(args), role=role)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            exact_task(("invented", "en", 0, False, 0, "LAB-00"))

    def test_incomplete_source_batch_never_reaches_training_sink(self):
        calls = []
        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=ValueError("stale source")):
            with self.assertRaises(ValueError):
                produce_two_step_train_candidates(
                    object(), SOURCE, lambda *x: calls.append(x), role=ROLE)
        self.assertEqual(calls, [])

    def test_duplicate_visible_frame_refused_before_sink(self):
        calls = []
        def fake_case(browser, url, scene):
            return [
                {"png": b"fake-only", "task": "invented",
                 "native_label": "invented", "rgb_digest": b"same"},
                {"png": b"fake-only", "task": "invented",
                 "native_label": "invented", "rgb_digest": b"same"},
            ]
        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=fake_case):
            with self.assertRaises(ValueError):
                produce_two_step_train_candidates(
                    object(), SOURCE, lambda *x: calls.append(x), role=ROLE)
        self.assertEqual(calls, [])

    def test_fake_completed_batch_never_claims_official_admission(self):
        calls = []
        seq = iter(range(32))
        def fake_case(browser, url, scene):
            return [
                {"png": b"fixture", "task": "fixture",
                 "native_label": "fixture", "rgb_digest": bytes([next(seq)])}
                for _ in range(2)
            ]
        with patch("h325.two_step_train_candidates.one_source_case",
                   side_effect=fake_case):
            result = produce_two_step_train_candidates(
                object(), SOURCE, lambda *x: calls.append(x), role=ROLE)
        self.assertEqual(result["train_candidate_messages"], 32)
        self.assertEqual(len(calls), 32)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["independent_dev_ood_qualified"])
        self.assertFalse(result["model_skill_gain_proven"])


@unittest.skipUnless(os.getenv("HALA_H325_BROWSER_QUALIFY") == "1",
                     "H325 original source browser opt-in required")
class H325RealBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.getenv("HALA_H325_BROWSER_CHANNEL", "chromium")
            if channel not in {"chromium", "msedge"}:
                raise ValueError("unsupported source-only browser")
            cls.browser = cls.pw.chromium.launch(
                headless=True,
                **({"channel": "msedge"} if channel == "msedge"
                   else {"args": ["--no-sandbox"]}))
        except Exception:
            cls.pw.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def test_all_16_real_two_step_native_scenarios(self):
        # This is a SOURCE GOLD extractor (not a model/policy): do not
        # print, save or publish any of the 32 training labels or images.
        count = [0]
        def sink(image, task, reference):
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("Hala", task)
            self.assertIn("actions", reference)
            count[0] += 1
        result = produce_two_step_train_candidates(
            self.browser, SOURCE, sink, role=ROLE)
        self.assertEqual(result["source_scenes"], 16)
        self.assertEqual(result["distinct_decoded_rgb_frames"], 32)
        self.assertEqual(result["original_native_action_states_verified"], 32)
        self.assertEqual(count[0], 32)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["model_skill_gain_proven"])


    def test_all_16_real_two_step_scenarios_one_shot(self):
        # Opt-in source-only qualification, never a model or persisted sink.
        calls = [0]

        def atomic_in_memory_sink(batch):
            calls[0] += 1
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 32)
            self.assertTrue(all(image.startswith(b"\\x89PNG\\r\\n\\x1a\\n")
                                for image, _, _ in batch))
            return True

        result = produce_two_step_train_candidates(
            self.browser, SOURCE, role=ROLE,
            trusted_batch_sink=atomic_in_memory_sink)
        self.assertEqual(calls[0], 1)
        self.assertEqual(result["candidate_handoff_mode"], "one_shot")
        self.assertEqual(result["original_native_action_states_verified"], 32)
        self.assertFalse(result["official_train_admitted"])


if __name__ == "__main__":
    unittest.main()

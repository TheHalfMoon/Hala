"""H338 original 128-scene source contracts and optional real browser.

Mocks create invented bytes ONLY to test atomic handoff; they never
qualify browser images or real Qwen/LoRA. Browser must qualify ALL scenes.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from h338.original_large_train_candidates import (
    ROLE, SCENES, IDS, SOURCE_BLOB, checked_source, exact_task,
    produce_large_train_candidates,
)

SOURCE = Path(__file__).resolve().parents[1] / "h338" / "signaldesk.html"


def fake_source_case(browser, url, scene):
    idx = SCENES.index(scene)
    return [
        {"png": bytes([idx, step]), "rgb_hash": sha256(
            bytes([idx, step])).digest(),
         "task": exact_task(scene),
         "native_label": "invented-only-synthetic-no-model"}
        for step in (0, 1)
    ]


class H338InventedOnlyContracts(unittest.TestCase):
    def test_128_frozen_bilingual_all_cartesian_combinations(self):
        self.assertEqual(len(SCENES), 128)
        self.assertEqual(len({scene[0] for scene in SCENES}), 128)
        self.assertEqual(len({scene[-1] for scene in SCENES}), 128)
        self.assertEqual(sum(scene[1] == "en" for scene in SCENES), 64)
        self.assertEqual(sum(scene[1] == "ar" for scene in SCENES), 64)
        combos = {(s[1], s[2], s[3], s[4], s[5]) for s in SCENES}
        self.assertEqual(len(combos), 128)
        self.assertTrue(all(len(exact_task(s)) < 500 for s in SCENES))
        self.assertEqual(len(IDS), 4)

    def test_source_pinned_to_exact_original_git_blob(self):
        import hashlib
        raw = SOURCE.read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()
        self.assertEqual(SOURCE_BLOB, blob)
        self.assertEqual(checked_source(SOURCE)[0], SOURCE.absolute())
        self.assertFalse(SOURCE.is_symlink())
        self.assertNotIn(b"https://", raw)
        self.assertNotIn(b"http://", raw)
        self.assertNotIn(b"<img", raw)
        self.assertNotIn(b"fetch(", raw)
        self.assertNotIn(b"XMLHttpRequest", raw)

    def test_frozen_roles_refuse_before_any_render(self):
        for role in (None, "", "TRAIN", "DEV", "BLIND_OOD",
                     "H325_ORIGINAL_TWO_STEP", "release"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                produce_large_train_candidates(
                    None, SOURCE, lambda x: True, role=role)
        with self.assertRaises(ValueError):
            produce_large_train_candidates(
                None, SOURCE, None, role=ROLE)

    def test_invented_handoff_is_one_shot_and_no_model_claim(self):
        calls = []
        def capture(batch):
            calls.append((len(batch), all(type(x) is tuple
                                          and len(x) == 3 for x in batch)))
            return True
        with patch(
            "h338.original_large_train_candidates.one_original_case",
            side_effect=fake_source_case,
        ) as original:
            result = produce_large_train_candidates(
                None, SOURCE, capture, role=ROLE)
        self.assertEqual(original.call_count, 128)
        self.assertEqual(calls, [(256, True)])
        self.assertEqual(result["source_scenes"], 128)
        self.assertEqual(result["distinct_decoded_rgb_images"], 256)
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["independent_dev_ood_qualified"])
        self.assertFalse(result["measured_model_improvement"])
        self.assertFalse(result["can_release"])

    def test_late_failure_must_not_emit_partial_sample(self):
        calls = []
        def flaky(browser, url, scene):
            if scene == SCENES[-1]:
                raise ValueError("REFUSE_FINAL_SCENE")
            return fake_source_case(browser, url, scene)
        with patch("h338.original_large_train_candidates.one_original_case",
                   side_effect=flaky), self.assertRaises(ValueError):
            produce_large_train_candidates(
                None, SOURCE, lambda batch: calls.append(batch),
                role=ROLE)
        self.assertEqual(calls, [])

    def test_duplicate_visible_frame_rejected_before_handoff(self):
        calls = []
        def duplicate(browser, url, scene):
            pair = fake_source_case(browser, url, scene)
            if scene == SCENES[-1]:
                pair[1]["rgb_hash"] = b"\x00" * 32
            if scene == SCENES[0]:
                pair[0]["rgb_hash"] = b"\x00" * 32
            return pair
        with patch("h338.original_large_train_candidates.one_original_case",
                   side_effect=duplicate), self.assertRaises(ValueError):
            produce_large_train_candidates(
                None, SOURCE, lambda batch: calls.append(batch),
                role=ROLE)
        self.assertEqual(calls, [])

    def test_sink_must_return_literal_true_not_truthy(self):
        with patch("h338.original_large_train_candidates.one_original_case",
                   side_effect=fake_source_case):
            for bad in (None, 1, "true", False):
                with self.subTest(returned=bad), self.assertRaises(ValueError):
                    produce_large_train_candidates(
                        None, SOURCE, lambda batch: bad, role=ROLE)


@unittest.skipUnless(os.environ.get("HALA_H338_BROWSER_QUALIFY") == "1",
                     "full actual source browser qualification requires opt-in")
class H338RealOriginal128(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H338_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("unsupported original source browser")
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

    def test_all_128_original_two_step_cases_and_256_unique_frames(self):
        witnessed = []
        def trusted_only(batch):
            # Data stays in RAM and never leaves this trusted test process.
            self.assertEqual(len(batch), 256)
            for image, task, action in batch:
                self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertIn("You are Hala", task)
                self.assertIn("actions", action)
            witnessed.append(len(batch))
            return True
        result = produce_large_train_candidates(
            self.browser, SOURCE, trusted_only, role=ROLE)
        self.assertEqual(witnessed, [256])
        self.assertEqual(result["source_scenes"], 128)
        self.assertEqual(result["validated_native_action_stages"], 256)
        self.assertEqual(result["distinct_decoded_rgb_images"], 256)
        self.assertEqual(result["english_scenes"], 64)
        self.assertEqual(result["arabic_rtl_scenes"], 64)
        self.assertFalse(result["official_train_admitted"])


if __name__ == "__main__":
    unittest.main()

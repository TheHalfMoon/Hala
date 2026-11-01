"""H342 invented-only contracts; opt-in real 128-scene source browser audit."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import importlib.util
import os
import unittest

from h342.original_visual_neighborhoods import (
    ROLE, FRAME_COUNT, _aggregate, _band, _fingerprint,
    audit_original_signaldesk, summarize_trusted_source_batch,
)

SOURCE = Path(__file__).resolve().parents[1] / "h338" / "signaldesk.html"


class H342InventedOnly(unittest.TestCase):
    def test_pair_accounting_with_invented_distinct_rgb_same_lossy_hash(self):
        invented = tuple((0, sha256(str(i).encode()).digest())
                         for i in range(FRAME_COUNT))
        result = _aggregate(invented)
        self.assertEqual(result["within_one_family_pairs"], 32640)
        self.assertEqual(
            result["perceptual_zero_distance_but_distinct_rgb_pairs"], 32640)
        self.assertEqual(result["near_0_to_32_within_language_pairs"], 16256)
        self.assertEqual(result["near_0_to_32_cross_language_pairs"], 16384)
        self.assertEqual(result["status"], ROLE)
        self.assertFalse(result["official_h199_train_admitted"])
        self.assertFalse(result["measured_model_gain"])
        self.assertFalse(result["can_release"])

    def test_invented_one_hot_hashes_have_distance_two(self):
        invented = tuple((1 << i, sha256(str(i).encode()).digest())
                         for i in range(FRAME_COUNT))
        report = _aggregate(invented)
        self.assertEqual(report["perceptual_hamming_distance_pair_counts"]["zero"], 0)
        self.assertEqual(report["perceptual_hamming_distance_pair_counts"]["1_to_8"], 32640)

    def test_invalid_counts_rgb_duplicates_and_bool_hash_rejected(self):
        invented = tuple((i, sha256(str(i).encode()).digest())
                         for i in range(FRAME_COUNT))
        for bad in (invented[:-1],
                    invented[:-1] + (invented[0],),
                    ((True, invented[0][1]),) + invented[1:],
                    ((1 << 1024, invented[0][1]),) + invented[1:]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                _aggregate(bad)

    def test_distance_band_boundaries_and_invalid(self):
        expected = {
            0: "zero", 1: "1_to_8", 8: "1_to_8",
            9: "9_to_32", 32: "9_to_32",
            33: "33_to_128", 128: "33_to_128",
            129: "129_to_1024", 1024: "129_to_1024",
        }
        for distance, name in expected.items():
            self.assertEqual(_band(distance), name)
        for invalid in (-1, 1025, True, 1.5):
            with self.assertRaises(ValueError):
                _band(invalid)

    def test_malformed_batch_rejected_before_decoding(self):
        for batch in (None, [], (), tuple([("not png", "task", "label")] * 256),
                      tuple([(b"x", "task", "label")] * 255)):
            with self.assertRaises(ValueError):
                summarize_trusted_source_batch(batch)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow optional for CPU")
    def test_real_decoded_invented_rgb_distinct_but_hash_alias(self):
        from PIL import Image
        from io import BytesIO
        images = []
        for shade in (40, 170):
            b = BytesIO()
            Image.new("RGB", (960, 720), (shade, shade, shade)).save(b, format="PNG")
            images.append(b.getvalue())
        a, b = (_fingerprint(raw) for raw in images)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        with self.assertRaises(ValueError):
            _fingerprint(b"not a PNG")

    def test_no_partial_report_if_source_validation_fails(self):
        with patch("h342.original_visual_neighborhoods.produce_large_train_candidates",
                   side_effect=ValueError("late source failure")):
            with self.assertRaises(ValueError):
                audit_original_signaldesk(None, SOURCE)

    def test_invented_sink_only_and_no_train_promotion(self):
        invented = tuple((b"fake", "invented task", "invented label")
                         for _ in range(256))
        def fake_produce(browser, source, sink, *, role):
            self.assertEqual(len(invented), 256)
            self.assertTrue(sink(invented))
            return {"source_scenes": 128, "validated_native_action_stages": 256,
                    "distinct_decoded_rgb_images": 256,
                    "official_train_admitted": False,
                    "measured_model_improvement": False}
        with patch("h342.original_visual_neighborhoods.produce_large_train_candidates",
                   side_effect=fake_produce), patch(
                   "h342.original_visual_neighborhoods.summarize_trusted_source_batch",
                   return_value={"status": ROLE, "can_release": False}):
            report = audit_original_signaldesk(None, SOURCE)
        self.assertEqual(report["status"], ROLE)
        self.assertFalse(report["can_release"])


@unittest.skipUnless(os.environ.get("HALA_H342_BROWSER_QUALIFY") == "1",
                     "real source browser audit requires opt-in")
class H342RealOriginalSource(unittest.TestCase):
    def test_all_128_scenes_256_frames_aggregate_only(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            channel = os.environ.get("HALA_H342_BROWSER_CHANNEL", "chromium")
            if channel not in ("chromium", "msedge"):
                raise ValueError("unsupported source browser")
            browser = pw.chromium.launch(
                headless=True, **({"channel": "msedge"} if channel == "msedge"
                                  else {"args": ["--no-sandbox"]}))
            try:
                report = audit_original_signaldesk(browser, SOURCE)
            finally:
                browser.close()
        self.assertEqual(report["source_frames"], 256)
        self.assertEqual(report["decoded_rgb_unique_frames"], 256)
        self.assertEqual(report["within_one_family_pairs"], 32640)
        self.assertEqual(sum(report["perceptual_hamming_distance_pair_counts"].values()), 32640)
        self.assertFalse(report["official_h199_train_admitted"])
        self.assertFalse(report["dev_or_blind_ood_qualified"])
        self.assertFalse(report["measured_model_gain"])


if __name__ == "__main__":
    unittest.main()

"""H326 source-owned Arabic RTL vs English coordinate audit, NOT model skill."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from h273.relative_native_bridge import normalized_to_pixel
from h326.rtl_relative_geometry_audit import (
    ROLE, audit_source_geometry, normalized_center,
)

SOURCE = Path(__file__).resolve().parents[1] / "h319" / "beaconboard.html"


class SourceGeometryContract(unittest.TestCase):
    def test_source_center_roundtrips_with_original_h273_mapping(self):
        for extent in (960, 720):
            for a, b in ((20, 160), (310, 260), (600, 90)):
                if a + b >= extent:
                    continue
                axis = "x" if extent == 960 else "y"
                box = {"x": a, "y": a, "width": b, "height": b}
                n = normalized_center(box, extent, axis)
                pixel = normalized_to_pixel(n, extent)
                self.assertGreater(pixel, a)
                self.assertLess(pixel, a + b)

    def test_invalid_control_geometries_fail_closed(self):
        for bad in (
            {"x": -1, "y": 0, "width": 100, "height": 60},
            {"x": 940, "y": 1, "width": 35, "height": 50},
            {"x": 12, "y": 12, "width": 3, "height": 30},
            {"x": "12", "y": 0, "width": 90, "height": 50},
            {"x": 0, "y": 0, "width": 100},
        ):
            with self.subTest(box=bad), self.assertRaises(ValueError):
                normalized_center(bad, 960, "x")

    def test_no_train_admission_or_blind_promotion(self):
        for role in ("TRAIN", "DEV", "BLIND_OOD", "H324"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                audit_source_geometry(None, SOURCE, role=role)

    def test_mocked_browser_success_cannot_claim_model_improvement(self):
        from h319.beaconboard_dynamic_research import SCENES
        def fake_source_case(browser, source, scene,
                             *, x_reflection_negative_control=False):
            return {
                "language": scene[1],
                "correct_focus_and_target_state":
                    not x_reflection_negative_control,
            }
        with patch("h326.rtl_relative_geometry_audit.source_case",
                   side_effect=fake_source_case):
            result = audit_source_geometry(None, SOURCE, role=ROLE)
        self.assertEqual(result["positive_source_cases_success"], 12)
        self.assertEqual(result["negative_controls_wrong_target"], 6)
        self.assertEqual(result["genuine_qwen_generations"], 0)
        self.assertFalse(result["adapted_gain_proven"])
        self.assertFalse(result["can_release"])

    def test_inversion_control_must_actually_fail(self):
        def fake_success_everywhere(browser, source, scene,
                                    *, x_reflection_negative_control=False):
            return {"language": scene[1],
                    "correct_focus_and_target_state": True}
        with patch("h326.rtl_relative_geometry_audit.source_case",
                   side_effect=fake_success_everywhere):
            with self.assertRaises(ValueError):
                audit_source_geometry(None, SOURCE, role=ROLE)


@unittest.skipUnless(os.environ.get("HALA_H326_BROWSER_QUALIFY") == "1",
                     "source-only original browser opt-in required")
class RealOriginalBrowserGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        try:
            channel = os.environ.get("HALA_H326_BROWSER_CHANNEL", "chromium")
            if channel not in {"chromium", "msedge"}:
                raise ValueError("unsupported source browser")
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

    def test_true_12_source_clicks_and_6_rtl_mirror_rejections(self):
        result = audit_source_geometry(self.browser, SOURCE, role=ROLE)
        self.assertEqual(result["positive_source_cases"], 12)
        self.assertEqual(result["positive_source_cases_success"], 12)
        self.assertEqual(result["arabic_rtl_positive_cases"], 6)
        self.assertEqual(result["negative_controls_wrong_target"], 6)
        self.assertEqual(result["genuine_qwen_generations"], 0)
        self.assertFalse(result["independent_blind_ood"])


if __name__ == "__main__":
    unittest.main()

"""H372: redacted, pinned original-source browser substage diagnostics.

All payloads are invented. Never output raw source tasks, images or exception
messages; these tests are not legal TRAIN, blind OOD or model evaluation.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from h319.beaconboard_dynamic_research import _observed_predict
from h370.browser_failure_stage import STAGES, browser_failure_code


class H372GranularSourceDiagnosticTests(unittest.TestCase):
    def test_stage_registry_is_fixed_and_bilingual_flows_safe(self):
        for flow in ("h319", "h337"):
            for stage in (
                "first_image_capture", "first_image_digest",
                "first_policy_callback", "first_post_policy_frame"
            ):
                with self.subTest(flow=flow, stage=stage):
                    self.assertIn(stage, STAGES[flow])
                    msg = browser_failure_code(
                        flow, stage, RuntimeError("SECRET PRIVATE GOLD"))
                    self.assertIn("stage=" + stage + " kind=runtime", msg)
                    self.assertNotIn("SECRET", msg)
                    self.assertNotIn("GOLD", msg)
        for stage in ("second_image_capture", "second_image_digest",
                      "second_policy_callback", "second_post_policy_frame"):
            self.assertIn(stage, STAGES["h319"])
            self.assertNotIn(stage, STAGES["h337"])

    def test_first_capture_failure_is_precisely_tagged(self):
        stage = []
        with patch(
            "h319.beaconboard_dynamic_research._observed_png",
            side_effect=RuntimeError("PRIVATE PHOTO")
        ):
            with self.assertRaises(RuntimeError):
                _observed_predict(
                    object(), lambda *_: "", "public", set(),
                    stage_sink=stage.append, phase="first")
        self.assertEqual(stage[-1], "first_image_capture")

    def test_first_policy_callback_failure_is_precisely_tagged(self):
        stage = []
        def fake_policy(*_):
            raise TimeoutError("SECRET DO NOT PRINT")
        with patch(
            "h319.beaconboard_dynamic_research._observed_png",
            return_value=b"\x89PNG\r\n\x1a\nsynthetic"
        ), patch(
            "h319.beaconboard_dynamic_research._model_visible_rgb_digest",
            return_value=b"fake invented RGB digest"
        ):
            with self.assertRaises(TimeoutError):
                _observed_predict(
                    type("FakePage", (), {"url": "file:///public"})(),
                    fake_policy, "synthetic public task", set(),
                    stage_sink=stage.append, phase="first")
        self.assertEqual(stage[-1], "first_policy_callback")

    def test_post_policy_screenshot_failure_is_distinguished(self):
        stage = []
        class FakePage:
            url = "file:///public-synthetic"
        calls = [b"\x89PNG\r\n\x1a\nsynthetic", TimeoutError("PRIVATE POST FRAME")]
        with patch(
            "h319.beaconboard_dynamic_research._observed_png",
            side_effect=calls
        ), patch(
            "h319.beaconboard_dynamic_research._model_visible_rgb_digest",
            return_value=b"fake RGB"
        ):
            with self.assertRaises(TimeoutError):
                _observed_predict(
                    FakePage(),
                    lambda *_: '{"actions":[{"kind":"abstain"}]}',
                    "public invented task", set(),
                    stage_sink=stage.append, phase="first")
        self.assertEqual(stage[-1], "first_post_policy_frame")

    def test_second_frame_uses_distinct_private_safe_label(self):
        stage = []
        with patch(
            "h319.beaconboard_dynamic_research._observed_png",
            side_effect=RuntimeError("PRIVATE SECOND SCREENSHOT")
        ):
            with self.assertRaises(RuntimeError):
                _observed_predict(
                    object(), lambda *_: "", "public", set(),
                    stage_sink=stage.append, phase="second")
        self.assertEqual(stage[-1], "second_image_capture")
        masked = browser_failure_code(
            "h319", stage[-1], RuntimeError("PRIVATE SECOND SCREENSHOT"))
        self.assertNotIn("PRIVATE", masked)


if __name__ == "__main__":
    unittest.main()

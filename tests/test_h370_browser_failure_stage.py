"""H370 source-only metadata redaction tests; invented errors, no browser."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import unittest

from h370.browser_failure_stage import STAGES, browser_failure_code
from h319.beaconboard_dynamic_research import (
    ROLE as ROLE319, SCENES, run_original_dynamic_sequence,
)
from h337.source_miss_distance_bands import (
    ROLE as ROLE337, source_first_click_distance,
)


class H370PrivateDataRedactionTests(unittest.TestCase):
    def test_both_flows_all_stages_are_fixed_and_bounded(self):
        self.assertEqual(set(STAGES), {"h319", "h337"})
        for flow, stages in STAGES.items():
            self.assertEqual(len(stages), len(set(stages)))
            for stage in stages:
                with self.subTest(flow=flow, stage=stage):
                    code = browser_failure_code(flow, stage, ValueError("SECRET"))
                    self.assertIn(" stage=" + stage + " kind=value", code)
                    self.assertNotIn("SECRET", code)
                    self.assertLess(len(code), 135)

    def test_never_interpolates_exception_text_or_custom_type_name(self):
        class SecretSceneCodeSIM21(Exception):
            pass
        for bad in (
            ValueError("task=SIM-21 image bytes points=(5,6)"),
            SecretSceneCodeSIM21("PRIVATE_GOLD_IMAGE"),
            OSError("D:/sensitive/path PRIVATE"),
            TimeoutError("browser screenshot=PRIVATE"),
            AssertionError("source-label B-11"),
        ):
            with self.subTest(kind=type(bad).__name__):
                code = browser_failure_code("h337", "first_render_and_policy", bad)
                self.assertTrue(code.startswith(
                    "H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED"))
                for leaked in ("SIM-21", "PRIVATE", "SECRET", "B-11",
                               "points=", "D:/", "SceneCode"):
                    self.assertNotIn(leaked, code)
                self.assertTrue(code.endswith(("kind=other", "kind=value",
                                                "kind=os", "kind=timeout",
                                                "kind=assertion")))

    def test_unrecognized_stage_or_flow_refused_without_payload(self):
        for flow, stage in (
            ("h337", "predicted_pixel_42"), ("private", "native_click"),
            ("h319", "PRIVATE_GOLD"), ("h337", ""),
        ):
            with self.subTest(flow=flow, stage=stage):
                with self.assertRaisesRegex(ValueError,
                                            "^H370 INVALID_DIAGNOSTIC_INPUT$"):
                    browser_failure_code(flow, stage, RuntimeError("SECRET"))

    def test_browser_exception_stage_survives_fail_closed_redaction(self):
        @contextmanager
        def fake_original_scene(*args):
            raise TimeoutError("SECRET TASK PRIVATE SCREENSHOT")
            yield None

        # Source identity is a mocked original-public resource, not real
        # private media. Only the bounded first-stage label can escape.
        with patch("h319.beaconboard_dynamic_research.checked_source",
                   return_value=(Path("beaconboard.html"), "file:///public")):
            with patch("h319.beaconboard_dynamic_research.original_scene",
                       fake_original_scene):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED "
                    r"stage=initial_original_scene kind=timeout$"
                ) as ctx:
                    run_original_dynamic_sequence(
                        None, Path("beaconboard.html"), SCENES[0],
                        lambda _png, _task: "", role=ROLE319)
                self.assertNotIn("SECRET", str(ctx.exception))

        with patch("h337.source_miss_distance_bands.checked_source",
                   return_value=(Path("beaconboard.html"), "file:///public")):
            with patch("h337.source_miss_distance_bands.original_scene",
                       fake_original_scene):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED "
                    r"stage=initial_original_scene kind=timeout$"
                ) as ctx:
                    source_first_click_distance(
                        None, Path("beaconboard.html"), SCENES[0],
                        lambda _png, _task: "", role=ROLE337)
                self.assertNotIn("SECRET", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

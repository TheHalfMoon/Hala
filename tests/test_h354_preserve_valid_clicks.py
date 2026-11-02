"""H354 invented-only CPU: retain valid native button clicks.

Source-only original scene unit test is separate from any genuine model run.
NO GPU or Qwen inference.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h354.preserve_valid_clicks import (
    preserve_valid_pixel_click, FROZEN_POLICY_SUFFIX,
)
from tests.test_h352_pixel_button_snap import invented_image, click
from h352.pixel_button_snap import pixel_button_rectangles
from h354.build_private_payload import build, SCRIPT, FLAG, REPORT, STATUS
from h354.verify_private_receipt import verify_success
from h353.build_private_payload import SOURCE_REV
from tests.test_h352_private_paired_snap import mock_pairs, HW
from h352.private_qwen_paired_snap import safe_report


class H354ConservativeTests(unittest.TestCase):
    def test_valid_pixel_inside_is_noop_for_both_themes(self):
        for dark in (False, True):
            png = invented_image(dark=dark)
            boxes = pixel_button_rectangles(png)
            self.assertEqual(len(boxes), 4)
            for box in boxes:
                left, top, right, bottom = box
                for point in ((left+3, top+4), (right-4, bottom-4),
                              ((left+right)//2, (top+bottom)//2)):
                    native = click(*point)
                    with self.subTest(dark=dark, point=point):
                        self.assertEqual(preserve_valid_pixel_click(png, native),
                                         native)

    def test_near_outside_is_relocated_when_unambiguous(self):
        for dark in (False, True):
            image = invented_image(dark=dark)
            original = click(438, 282)
            revised = preserve_valid_pixel_click(image, original)
            self.assertNotEqual(revised, original)
            self.assertEqual(
                json.loads(revised)["actions"][0]["point"],
                json.loads(__import__(
                    "h352.pixel_button_snap", fromlist=["snap_native_first_click"]
                ).snap_native_first_click(image, original))["actions"][0]["point"])

    def test_far_or_abstained_does_not_change(self):
        png = invented_image()
        for original in (click(940, 660),
                         '{"actions":[{"kind":"abstain"}]}'):
            self.assertEqual(preserve_valid_pixel_click(png, original), original)

    def test_invalid_action_and_image_refuses(self):
        with self.assertRaises(ValueError):
            preserve_valid_pixel_click(b"not image", click(200, 280))
        with self.assertRaises(ValueError):
            preserve_valid_pixel_click(invented_image(), '{"actions":[]}')
        with self.assertRaises(ValueError):
            preserve_valid_pixel_click(invented_image(), 'a'*257)

    def test_embedded_frozen_policy_matches_local_wrapper(self):
        from h353.build_private_payload import _pinned_file
        source = _pinned_file(Path(".").resolve(),
                              "h352/pixel_button_snap.py")
        policy = source + FROZEN_POLICY_SUFFIX
        namespace = {"__name__": "h354_embedded_source"}
        exec(compile(policy, "h354_pinned_policy", "exec"), namespace)
        snap = namespace["snap_native_first_click"]
        for dark in (False, True):
            png = invented_image(dark=dark)
            for original in (click(438, 282), click(100, 275),
                             click(940, 660),
                             '{"actions":[{"kind":"abstain"}]}'):
                with self.subTest(dark=dark, original=original):
                    self.assertEqual(snap(png, original),
                                     preserve_valid_pixel_click(png, original))

    def test_private_bundle_exact_byte_identity_and_no_media(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)/"private"
            r = build(Path(".").resolve(), out)
            self.assertFalse(r["gpu_submitted"])
            self.assertEqual(r["media_gold_weights"], 0)
            self.assertEqual(r["files"], 2)
            self.assertEqual(r["parent_source_revision"], SOURCE_REV)
            raw = (out/SCRIPT).read_bytes()
            self.assertEqual(r["uploaded_code_sha256"], sha256(raw).hexdigest())
            script = raw.decode("utf-8")
            self.assertIn(FLAG, script)
            self.assertIn(REPORT, script)
            self.assertIn(STATUS, script)
            self.assertIn("def paired_arm_order(scene):", script)
            self.assertIn("if any(left <= x < right and top <= y < bottom",script)
            self.assertNotIn("H354_SNAP_SOURCE_PENDING",script)
            metadata = json.loads((out/"kernel-metadata.json").read_text())
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"],[])
            self.assertEqual(metadata["model_sources"],[])
            with self.assertRaises(ValueError):
                build(Path(".").resolve(), out)

    def test_aggregate_receipt_uses_same_strict_contract(self):
        source = safe_report(mock_pairs(), HW)
        source["status"] = STATUS
        verified = verify_success(source)
        self.assertEqual(verified["by_language"]["ar"]["paired_wins"],2)
        self.assertFalse(verified["new_trained_model"])
        self.assertFalse(verified["can_release"])
        bad = dict(source)
        bad["source_private_point"] = [25,40]
        with self.assertRaises(ValueError):
            verify_success(bad)


if __name__ == "__main__":
    unittest.main()

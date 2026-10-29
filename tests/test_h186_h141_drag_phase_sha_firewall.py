"""H186: old H141 original drag-phase screenshots must stay excluded.

Original source support:
- h141/interactive_source.py: at_rest_image+at_rest_sha256,
  during_drag_image+during_drag_sha256, explicitly diagnostic-only.
- h141/audit.py: verifies these two old PNG paths/hashes, not RGBA.
- h141/audit_visible.py: all 16 previous frame digests excluded.
All cases here are synthetic; no historical images, gold or model data.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields,
    screen_unadmitted_new_case_hashes,
)
from h185.audit_manifest_hash_coverage import run as old_cli
from h186.audit_old_drag_png_aliases import run as h186_cli


def fixture(root: Path) -> None:
    phase = root / "private-h141" / "old-original-metadata"
    phase.mkdir(parents=True)
    (phase / "manifest.json").write_text(json.dumps({
        "private_cases": [
            {"at_rest_sha256": "1"*64, "at_rest_image": "old-private-rest.png",
             "during_drag_sha256": "2"*64,
             "during_drag_image": "old-private-in-motion.png",
             "gold_bbox": [10, 20, 30, 40]},
            {"image_sha256": "3"*64,
             "model_sha256": "4"*64,
             "license_sha256": "5"*64,
             "sha256": "6"*64,
             "rgba_sha256": "a"*64},
        ],
        "never_training": True,
    }), encoding="utf8")


class H186H141LegacyDragPhasePNGTests(unittest.TestCase):
    def test_two_old_h141_drag_png_families_recognized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            got = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(got["distinct_png_sha_candidates"], 3)
            self.assertEqual(got["h141_at_rest_png_sha_occurrences"], 1)
            self.assertEqual(got["h141_during_drag_png_sha_occurrences"], 1)
            self.assertEqual(got["distinct_old_drag_phase_png_sha_candidates"], 2)
            self.assertEqual(got["_owner_only_png_sha_candidates"],
                             frozenset({"1"*64, "2"*64, "3"*64}))
            self.assertEqual(got["_owner_only_rgba_sha_candidates"],
                             frozenset({"a"*64}))
            self.assertFalse(got["training_dataset_admitted"])
            self.assertFalse(got["complete_historical_manifest_coverage_proven"])

    def test_old_at_rest_image_sha_is_excluded_even_with_new_png_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            hist = scan_legacy_manifest_sha_fields([root])
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                screen_unadmitted_new_case_hashes([
                    {"image_sha256": "1"*64, "visual_rgba_sha256": "b"*64}
                ], hist)

    def test_old_during_drag_image_sha_is_excluded_even_with_new_png_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            hist = scan_legacy_manifest_sha_fields([root])
            with self.assertRaisesRegex(ValueError, "reuses known old PNG"):
                screen_unadmitted_new_case_hashes([
                    {"image_sha256": "2"*64, "visual_rgba_sha256": "b"*64}
                ], hist)

    def test_unrelated_model_license_generic_shas_not_counted_as_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            hist = scan_legacy_manifest_sha_fields([root])
            for illegal in ("4"*64, "5"*64, "6"*64):
                with self.subTest(illegal=illegal[:1]):
                    self.assertNotIn(illegal, hist["_owner_only_png_sha_candidates"])

    def test_only_new_hashes_are_not_formal_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            hist = scan_legacy_manifest_sha_fields([root])
            proof = screen_unadmitted_new_case_hashes([
                {"image_sha256": "7"*64, "visual_rgba_sha256": "b"*64}
            ], hist)
            self.assertFalse(proof["historical_completeness_proven"])
            self.assertFalse(proof["training_approved"])
            self.assertFalse(proof["ood_evaluation_approved"])
            self.assertEqual(proof["model_calls"], 0)

    def test_old_diagnostic_gold_and_images_never_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            data = root / "private-h141" / "old-original-metadata"
            image = data / "old-private-rest.png"
            image.write_bytes(b"do not open old screenshot")
            orig = Path.read_bytes
            opened = []
            def check(path):
                opened.append(path.name)
                if path.name != "manifest.json":
                    raise AssertionError("Old media or gold source read attempted")
                return orig(path)
            with patch.object(Path, "read_bytes", check):
                proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(opened, ["manifest.json"])
            self.assertFalse(proof["gold_field_values_parsed"])
            self.assertFalse(proof["old_image_media_opened"])

    def test_h186_cli_reports_only_aggregate_old_frame_alias_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = h186_cli(["--historical-root", str(root)])
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertIn("h141_at_rest_occurrences=1", out.getvalue())
            self.assertIn("h141_during_drag_occurrences=1", out.getvalue())
            self.assertIn("unique_h141_drag_phase_png_hashes=2", out.getvalue())
            self.assertIn("old_png_hashes=3", out.getvalue())
            self.assertIn("old_rgba_hashes=1", out.getvalue())
            for value in (str(root), "1"*64, "2"*64, "3"*64,
                          "old-private-rest.png", "gold_bbox"):
                self.assertNotIn(value, out.getvalue())

    def test_owner_cli_never_leaks_old_values_paths_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = old_cli(["--historical-root", str(root)])
            self.assertEqual(code, 0)
            for secret in ("1"*64, "2"*64, str(root),
                           "old-private-rest.png", "gold_bbox"):
                self.assertNotIn(secret, out.getvalue())
            self.assertIn("unique_png_sha_candidates=3", out.getvalue())
            self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()

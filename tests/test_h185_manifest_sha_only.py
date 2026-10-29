"""H185 synthetic manifest SHA-only coverage and privacy regression suite."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h185.audit_manifest_hash_coverage import run as cli_run


def _manifest(root: Path, name: str, data: dict, *,
              subfolder: str = "private-provenance") -> Path:
    folder = root / name / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "manifest.json"
    target.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return target


class LegacyManifestSHAOnlyTests(unittest.TestCase):
    def test_strict_sha_fields_extract_only_not_old_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = _manifest(root, "private-h139", {
                "cases": [{
                    "image_sha256": "1"*64,
                    "visual_rgba_sha256": "a"*64,
                    "gold_bbox": [3, 5, 50, 60],
                    "owner_private_model_prediction": "secret-target",
                }],
                "model_sha256": "9"*64,
                "gold": "private",
            })
            result = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(result["manifest_files_scanned"], 1)
            self.assertEqual(result["distinct_png_sha_candidates"], 1)
            self.assertEqual(result["distinct_decoded_rgba_sha_candidates"], 1)
            self.assertEqual(result["_owner_only_png_sha_candidates"],
                             frozenset({"1"*64}))
            self.assertEqual(result["_owner_only_rgba_sha_candidates"],
                             frozenset({"a"*64}))
            self.assertFalse(result["historical_gold_accessed"])
            self.assertFalse(result["old_image_media_opened"])
            self.assertFalse(result["training_dataset_admitted"])

    def test_multiple_old_phase_manifests_dedup_only_digest_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest(root, "private-h139", {
                "cases": [
                    {"image_sha256": "1"*64, "rgba_sha256": "a"*64},
                    {"png_sha256": "2"*64, "rgba_sha256": "b"*64},
                ]
            })
            _manifest(root, "private-h140", {
                "cases": [
                    {"image_sha256": "1"*64, "visual_rgba_sha256": "a"*64}
                ]
            })
            proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(proof["historical_phases_enumerated"], 2)
            self.assertEqual(proof["manifest_files_scanned"], 2)
            self.assertEqual(proof["png_sha_occurrences"], 3)
            self.assertEqual(proof["decoded_rgba_sha_occurrences"], 3)
            self.assertEqual(proof["distinct_png_sha_candidates"], 2)
            self.assertEqual(proof["distinct_decoded_rgba_sha_candidates"], 2)
            self.assertFalse(proof["complete_historical_manifest_coverage_proven"])
            self.assertFalse(proof["old_image_integrity_proven"])

    def test_missing_rgba_is_reported_not_misrepresented_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest(root, "private-h139", {
                "image_sha256": "2"*64,
                "image_path": "../private-ood-original.png",
                "model_weights_sha256": "b"*64,
            })
            proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(proof["manifests_with_decoded_rgba_sha_fields"], 0)
            self.assertEqual(proof["distinct_decoded_rgba_sha_candidates"], 0)
            self.assertFalse(proof["complete_historical_manifest_coverage_proven"])

    def test_non_manifest_files_never_opened_or_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            included = _manifest(root, "private-h139", {
                "image_sha256": "1"*64,
                "visual_rgba_sha256": "a"*64,
            })
            ignored = included.parent / "private-prediction.json"
            ignored.write_text('{"image_sha256":"' + "2"*64 + '"}')
            image = included.parent / "legacy-ood.png"
            image.write_bytes(b"\x89PNG private NEVER READ")
            original = Path.read_bytes
            calls = []
            def reading(path):
                calls.append(path.name)
                if path.name != "manifest.json":
                    raise AssertionError("Old image or prediction read attempted")
                return original(path)
            with patch.object(Path, "read_bytes", reading):
                proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(calls, ["manifest.json"])
            self.assertEqual(proof["distinct_png_sha_candidates"], 1)
            self.assertEqual(proof["model_calls"], 0)

    def test_paths_in_ignored_data_directories_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = _manifest(root, "private-h139", {
                "image_sha256": "1"*64,
            })
            hidden = root / "private-h139" / "gold" / "manifest.json"
            hidden.parent.mkdir(parents=True)
            hidden.write_text('{"image_sha256":"' + "2"*64 + '"}')
            proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(proof["manifest_files_scanned"], 1)
            self.assertEqual(proof["_owner_only_png_sha_candidates"],
                             frozenset({"1"*64}))

    def test_legacy_sha_values_not_printed_or_private_paths_echoed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest(root, "private-h139", {
                "image_sha256": "1"*64,
                "visual_rgba_sha256": "a"*64,
            })
            output, error = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                code = cli_run(["--historical-root", str(root)])
            self.assertEqual(code, 0)
            self.assertIn("unique_png_sha_candidates=1", output.getvalue())
            self.assertIn("unique_rgba_sha_candidates=1", output.getvalue())
            self.assertIn("H185_HISTORY_COMPLETENESS_UNPROVEN", output.getvalue())
            self.assertNotIn("1"*64, output.getvalue())
            self.assertNotIn("a"*64, output.getvalue())
            self.assertNotIn(str(root), output.getvalue())
            self.assertEqual(error.getvalue(), "")

    def test_non_sha_fields_and_malformed_sha_values_cannot_be_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest(root, "private-h139", {
                "model_sha256": "f"*64,
                "image_sha256": "UPPERCASE-SHA-NOT-VALID",
                "source_git_sha256": "b"*64,
                "visual_rgba_sha256": "c"*63,
            })
            proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(proof["distinct_png_sha_candidates"], 0)
            self.assertEqual(proof["distinct_decoded_rgba_sha_candidates"], 0)
            self.assertFalse(proof["training_dataset_admitted"])

    def test_missing_manifest_or_invalid_owner_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "private-h139").mkdir()
            with self.assertRaisesRegex(ValueError, "no allowlisted"):
                scan_legacy_manifest_sha_fields([root])
            with self.assertRaises(ValueError):
                scan_legacy_manifest_sha_fields([])
            with self.assertRaises(ValueError):
                scan_legacy_manifest_sha_fields([str(root / "not-found")])

    def test_refuse_symlinked_manifest_not_read_or_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = _manifest(root, "private-h139", {
                "image_sha256": "1"*64
            })
            linked = root / "private-h140" / "old-archive" / "manifest.json"
            linked.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(str(target), str(linked))
            except (OSError, NotImplementedError):
                self.skipTest("Windows cannot create a test symlink")
            with self.assertRaisesRegex(ValueError, "unsafe manifest file"):
                scan_legacy_manifest_sha_fields([root])

    def test_ignored_gold_or_model_symlink_subtrees_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _manifest(root, "private-h139", {"image_sha256": "1"*64})
            linked_dir = root / "private-h139" / "gold"
            try:
                os.symlink(str(root / "private-h139" / "private-provenance"),
                           str(linked_dir), target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Windows cannot create a test symlink")
            proof = scan_legacy_manifest_sha_fields([root])
            self.assertEqual(proof["manifest_files_scanned"], 1)


if __name__ == "__main__":
    unittest.main()

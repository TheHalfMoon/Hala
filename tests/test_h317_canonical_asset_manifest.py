"""H317: invented-only canonical metadata and duplicate-key attack cases.

No actual source app, asset, private content or training sample is inspected.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from h315.asset_custody_audit import audit, _strict_json_object

_BASE = {
    "source_repo": "fictional/fixture",
    "source_commit": "a" * 40,
    "asset_path": "assets/icon.svg",
    "asset_sha256": "sha256:" + "b" * 64,
    "license_spdx": "OFL-1.1",
    "license_evidence_commit": "c" * 40,
    "author_attestation": True,
    "rendered_digest": "sha256:" + "d" * 64,
    "historical_provenance_review": True,
    "independent_rights_review": True,
}


class CanonicalMetadataTests(unittest.TestCase):
    def test_normal_paths_still_quarantined(self):
        for path in ("assets/icon.svg", "Arabic/أيقونة.png", "icons/logo@2x.png"):
            with self.subTest(path=path):
                item = dict(_BASE, asset_path=path)
                result = audit({"assets": [item]})
                self.assertEqual(result["decision"], "NO_ADMISSION")
                self.assertEqual(result["admitted_count"], 0)

    def test_path_escape_and_aliases_refused(self):
        bad_paths = (
            "/abs/icon.svg", "../icon.svg", "a/../../icon.svg",
            "a/./icon.svg", "a//icon.svg", "a/", "a\\icon.svg",
            "C:\\icons\\icon.svg", "https://site.invalid/asset.svg",
            "a/%2e%2e/icon.svg", "a/secret?view=1",
            "a/secret#fragment", "a/secret\x00.png",
            "a/secret\u200b.png", "a/../b.png", "a/file. ",
        )
        for path in bad_paths:
            with self.subTest(path=repr(path)), self.assertRaises(ValueError):
                audit({"assets": [dict(_BASE, asset_path=path)]})

    def test_source_repository_rejects_ambiguous_refs(self):
        bad_repos = (
            "https://github.com/owner/repo", "owner/repo/extra",
            "owner", "../repo", "owner/..", "owner/repo.",
            "owner\\repo", "owner/repo?ref=main", "owner/repo\nsecret",
        )
        for repo in bad_repos:
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                audit({"assets": [dict(_BASE, source_repo=repo)]})

    def test_duplicate_json_envelope_keys_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            json.loads('{"assets":[],"assets":[]}',
                       object_pairs_hook=_strict_json_object)

    def test_duplicate_nested_keys_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            json.loads('{"assets":[{"source_repo":"a/b","source_repo":"c/d"}]}',
                       object_pairs_hook=_strict_json_object)

    def test_cli_fails_closed_without_echoing_untrusted_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            path.write_text(
                '{"assets":[],"assets":[],"secret":"DO_NOT_ECHO_PRIVATE"}',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "h315.asset_custody_audit",
                 "--manifest", str(path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("INVALID_METADATA", result.stdout)
            self.assertNotIn("DO_NOT_ECHO_PRIVATE", result.stdout + result.stderr)
            self.assertNotIn(str(path), result.stdout)


if __name__ == "__main__":
    unittest.main()

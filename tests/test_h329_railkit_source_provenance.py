"""H329: source pin + Git provenance contract. No private media/model."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from h329.railkit_source_provenance import (
    ORIGIN, ORIGIN_PARENT, SOURCE_BLOB, LICENSE_BLOB, ROLE,
    audit_original_html, audit_git_history,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h325" / "railkit.html"


class H329SourceEvidence(unittest.TestCase):
    def test_original_source_bytes_have_no_external_asset_reference(self):
        report = audit_original_html(SOURCE.read_bytes())
        self.assertEqual(report["external_html_assets_or_links"], 0)
        self.assertEqual(report["external_css_imports_or_urls"], 0)
        self.assertEqual(report["dynamic_script_network_loader_patterns"], 0)
        self.assertEqual(report["inline_style_count"], 1)
        self.assertEqual(report["inline_script_count"], 1)
        self.assertEqual(len(SOURCE_BLOB), 40)
        self.assertEqual(len(LICENSE_BLOB), 40)

    def test_modified_source_rejected(self):
        original = SOURCE.read_bytes()
        for mutated in (original.replace(b"RailKit", b"RailBad"),
                        original + b"<img src='https://example.org/asset.png'>",
                        original + b"<style>@import url(https://x)</style>",
                        original + b"<script>fetch('https://x')</script>",
                        b"not a source"):
            with self.subTest(length=len(mutated)), self.assertRaises(ValueError):
                audit_original_html(mutated)

    def test_inadmissible_git_head_rejected_without_git_call(self):
        for wrong in ("main", ORIGIN[:12], "z"*40, "a"*41, True, ""):
            with self.subTest(head=wrong), patch(
                "h329.railkit_source_provenance._git",
                side_effect=AssertionError("Should not touch Git"),
            ):
                with self.assertRaises(ValueError):
                    audit_git_history(ROOT, expected_head=wrong)

    def test_pinned_origin_parent_and_source_hash_contract(self):
        self.assertEqual(ORIGIN, "3303396b77dc7671cd97f425c644f71b6b0b664e")
        self.assertEqual(ORIGIN_PARENT, "9c636b7c73679d8422db13a15e3fe7a298b39606")
        self.assertEqual(SOURCE_BLOB, "743b5e1b54774a1deea4dbe6bfbe0613adead7e4")
        self.assertEqual(LICENSE_BLOB, "d645695673349e3947e8e5ae42332d0ac3164cd7")

    def test_false_rights_claims_never_made(self):
        self.assertIn("NOT_RIGHTS_ADMISSION", ROLE)
        with patch("h329.railkit_source_provenance._git",
                   side_effect=ValueError("Git unavailable")):
            with self.assertRaises(ValueError):
                audit_git_history(ROOT, expected_head=ORIGIN)


@unittest.skipUnless(os.environ.get("HALA_H329_GIT_QUALIFY") == "1",
                     "full-history real Git provenance opt-in required")
class H329RealPinnedHistory(unittest.TestCase):
    def test_true_original_git_history_and_rights_nonadmission(self):
        import subprocess
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True,
            capture_output=True).stdout.strip()
        report = audit_git_history(ROOT, expected_head=sha)
        self.assertEqual(report["source_pinned_git_blob"], True)
        self.assertTrue(report["source_new_in_origin_parent"])
        self.assertTrue(report["origin_is_ancestor_of_checked_head"])
        self.assertTrue(report["apache_2_license_blob_unchanged_from_origin"])
        self.assertFalse(report["independent_copyright_title_reviewed"])
        self.assertFalse(report["generated_screenshot_license_scope_independently_reviewed"])
        self.assertFalse(report["legal_train_use_authorized"])
        self.assertFalse(report["official_h199_train_admitted"])
        self.assertFalse(report["can_release"])
        self.assertEqual(report["model_generations"], 0)


if __name__ == "__main__":
    unittest.main()

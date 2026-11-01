"""H339 source-only original custody and legal nonadmission tests.

A Git introduction, Apache LICENSE and no embedded media are evidence,
but NEVER independently establish screenshot ML training rights.
"""
from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest.mock import patch

from h339.original_source_rights_evidence import (
    ORIGIN, ORIGIN_PARENT, ROLE, SOURCE_BLOB, LICENSE_BLOB,
    audit_signaldesk_html, audit_git_history, combined_candidacy_report,
    audit_both_sources,
)
from h329.railkit_source_provenance import ROLE as RAIL_ROLE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "h338" / "signaldesk.html"


def trusted_source_shapes() -> tuple[dict, dict]:
    rail = {
        "status": RAIL_ROLE,
        "origin_commit": "3303396b77dc7671cd97f425c644f71b6b0b664e",
        "source_pinned_git_blob": True,
        "source_new_in_origin_parent": True,
        "origin_is_ancestor_of_checked_head": True,
        "apache_2_license_blob_unchanged_from_origin": True,
        "external_html_assets_or_links": 0,
        "external_css_imports_or_urls": 0,
        "dynamic_script_network_loader_patterns": 0,
        "inline_script_count": 1,
        "inline_style_count": 1,
        "verified_github_commit_cryptographic_signature": False,
        "complete_external_source_history_proven": False,
        "model_generations": 0, "optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "independent_copyright_title_reviewed": False,
        "generated_screenshot_license_scope_independently_reviewed": False,
        "legal_train_use_authorized": False,
        "official_h199_train_admitted": False,
        "independent_dev_or_blind_ood_qualified": False,
        "can_release": False,
    }
    signal = {
        "status": ROLE, "introduction_commit": ORIGIN,
        "pinned_original_html_git_blob": SOURCE_BLOB,
        "previous_parent_lacks_source": True,
        "introduction_is_ancestor_of_current_commit": True,
        "unchanged_apache_2_license_git_blob": True,
        "inline_script_count": 1, "inline_style_count": 1,
        "external_css_resources": 0,
        "source_author_or_title_independently_proven": False,
        "introduction_commit_cryptographically_signed": False,
        "adapted_model_gain_proven": False,
        "model_generations": 0, "optimizer_steps": 0,
        "external_html_assets_or_links": 0,
        "dynamic_network_loaders": 0,
        "full_external_copyright_title_proven": False,
        "generated_screenshot_training_rights_independently_granted": False,
        "independent_source_family_rights_review_completed": False,
        "official_h199_train_admitted": False,
        "dev_or_blind_ood_qualified": False,
        "can_release": False,
    }
    return rail, signal


class H339OriginalSourceContracts(unittest.TestCase):
    def test_source_blob_inline_only_rights_still_false(self):
        r = audit_signaldesk_html(SOURCE.read_bytes())
        self.assertEqual(r["pinned_original_html_git_blob"], SOURCE_BLOB)
        self.assertEqual(r["inline_style_count"], 1)
        self.assertEqual(r["inline_script_count"], 1)
        self.assertEqual(r["external_html_assets_or_links"], 0)
        self.assertEqual(r["dynamic_network_loaders"], 0)
        self.assertFalse(r["source_author_or_title_independently_proven"])
        self.assertEqual(len(LICENSE_BLOB), 40)

    def test_modified_source_remote_assets_refused(self):
        original = SOURCE.read_bytes()
        for changed in (
            original.replace(b"SignalDesk", b"SignalWrong"),
            original + b'<img src="https://example.com/record.png">',
            original.replace(b"</style>", b"@import url(https://x);</style>"),
            original.replace(b"</script>", b"fetch('/api');</script>"),
            original + b'<script>new WebSocket("wss://x")</script>',
            original + b"<iframe src='x'></iframe>",
            b"wrong file",
        ):
            with self.subTest(size=len(changed)), self.assertRaises(ValueError):
                audit_signaldesk_html(changed)

    def test_invalid_commit_ref_rejected_without_git(self):
        for head in ("main", ORIGIN[:12], "z"*40, "f"*41, True, ""):
            with self.subTest(head=head), patch(
                "h339.original_source_rights_evidence._git",
                side_effect=AssertionError("git should not run"),
            ), self.assertRaises(ValueError):
                audit_git_history(ROOT, expected_head=head)

    def test_combined_source_counts_no_rights_or_model_claim(self):
        rail, signal = trusted_source_shapes()
        result = combined_candidacy_report(rail, signal)
        self.assertEqual(result["candidate_source_families"], 2)
        self.assertEqual(result["total_candidate_scenes"], 144)
        self.assertEqual(result["total_candidate_action_messages"], 288)
        self.assertEqual(result["english_candidate_scenes"], 72)
        self.assertEqual(result["arabic_rtl_candidate_scenes"], 72)
        self.assertEqual(result["independent_rights_reviewer_approvals"], 0)
        self.assertFalse(result["can_use_as_training_data"])
        self.assertFalse(result["independent_dev_or_blind_ood_qualified"])
        self.assertFalse(result["adapted_model_gain_proven"])
        self.assertFalse(result["can_release"])

    def test_forged_training_rights_or_dev_claim_rejected(self):
        for side, key, value in (
            ("rail", "legal_train_use_authorized", True),
            ("rail", "origin_is_ancestor_of_checked_head", False),
            ("rail", "model_generations", 1),
            ("rail", "dynamic_script_network_loader_patterns", 1),
            ("signal", "model_generations", 1),
            ("signal", "optimizer_steps", 1),
            ("signal", "introduction_commit_cryptographically_signed", True),
            ("rail", "official_h199_train_admitted", True),
            ("signal", "full_external_copyright_title_proven", True),
            ("signal", "generated_screenshot_training_rights_independently_granted", True),
            ("signal", "official_h199_train_admitted", True),
            ("signal", "dev_or_blind_ood_qualified", True),
            ("signal", "can_release", True),
            ("signal", "pinned_original_html_git_blob", "0"*40),
            ("signal", "dynamic_network_loaders", 1),
        ):
            rail, signal = trusted_source_shapes()
            target = rail if side == "rail" else signal
            target[key] = value
            with self.subTest(field=(side,key)), self.assertRaises(ValueError):
                combined_candidacy_report(rail, signal)

    def test_wrong_origin_parent_and_source_rights_not_promoted(self):
        self.assertEqual(
            ORIGIN, "ae183d4531ccf1ada8c98098416e38e715332cd6")
        self.assertEqual(
            ORIGIN_PARENT, "5aca7ceb8e4e5d86d736dd83683a60ef84d2d3f2")
        with patch(
            "h339.original_source_rights_evidence._git",
            side_effect=ValueError("Git unavailable"),
        ), self.assertRaises(ValueError):
            audit_git_history(ROOT, expected_head=ORIGIN)

    def test_both_sources_require_real_git_and_complete_review(self):
        with patch(
            "h339.original_source_rights_evidence.audit_railkit_git_history",
            side_effect=ValueError("not all git source history"),
        ), self.assertRaises(ValueError):
            audit_both_sources(ROOT, expected_head=ORIGIN)


@unittest.skipUnless(os.environ.get("HALA_H339_GIT_QUALIFY") == "1",
                     "full Git history and unchanged source binaries required")
class H339RealGitCustody(unittest.TestCase):
    def test_original_signaldesk_git_origin_and_combined_nonadmission(self):
        import subprocess
        head = subprocess.run(
            ["git","rev-parse","HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        result = audit_git_history(ROOT, expected_head=head)
        self.assertTrue(result["previous_parent_lacks_source"])
        self.assertTrue(result["introduction_is_ancestor_of_current_commit"])
        self.assertTrue(result["unchanged_apache_2_license_git_blob"])
        self.assertFalse(result["introduction_commit_cryptographically_signed"])
        self.assertFalse(result["generated_screenshot_training_rights_independently_granted"])
        self.assertFalse(result["official_h199_train_admitted"])
        combined = audit_both_sources(ROOT, expected_head=head)
        self.assertEqual(combined["total_candidate_action_messages"], 288)
        self.assertEqual(combined["independent_rights_reviewer_approvals"], 0)
        self.assertFalse(combined["can_use_as_training_data"])


if __name__ == "__main__":
    unittest.main()

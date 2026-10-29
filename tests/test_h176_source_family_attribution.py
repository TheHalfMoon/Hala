"""H176 source-family attribution: strictly synthetic metadata-only tests.

All hashes and pseudo-owner repositories in these tests are fabricated except
the already-published H173 immutable source pins. Never load consumed media.
"""
from copy import deepcopy
import unittest

from h173.source_preflight import SOURCE_PINS
from h174.source_history_gate import SCHEMA as HISTORY_SCHEMA
from h174.split_firewall import (
    BANNED, HISTORY_SCHEMA as PIXEL_HISTORY_SCHEMA, SCHEMA as SPLIT_SCHEMA,
)
from h176.source_family_attribution import SCHEMA, audit_attribution


def _fixture():
    metadata = {
        "schema": SPLIT_SCHEMA,
        "sources": [
            {"app": "freshclick", "split": "TRAIN", "repo": "newowner/click",
             "git_sha": "a"*40, "html_sha256": "a"*64, "license": "MIT"},
            {"app": "freshdrag", "split": "DEV", "repo": "newowner/drag",
             "git_sha": "b"*40, "html_sha256": "b"*64, "license": "MIT"},
            {"app": "freshood", "split": "OOD_RESERVED", "repo": "newowner/ood",
             "git_sha": "c"*40, "html_sha256": "c"*64, "license": "MIT"},
        ],
        "cases": [
            {"index": 11, "app": "freshclick", "split": "TRAIN",
             "image_sha256": "1"*64, "visual_rgba_sha256": "2"*64,
             "language": "en", "action_type": "click",
             "instruction": "Click the original button"},
            {"index": 12, "app": "freshdrag", "split": "DEV",
             "image_sha256": "3"*64, "visual_rgba_sha256": "4"*64,
             "language": "ar", "action_type": "drag",
             "instruction": "اسحب العنصر إلى الهدف"},
        ],
    }
    pixels = {
        "schema": PIXEL_HISTORY_SCHEMA, "image_shas": ["8"*64],
        "visual_shas": ["9"*64],
    }
    rows = []
    for index, app in enumerate(sorted(BANNED), start=1):
        frozen = SOURCE_PINS.get(app)
        rows.append({
            "app": app, "repo": f"fictional-owner/old-original-{index}",
            "git_sha": frozen["git_sha"] if frozen else f"{index:040x}",
            "html_sha256": (frozen["original_html_sha256"] if frozen
                            else f"{index:064x}"),
        })
    history = {
        "schema": HISTORY_SCHEMA,
        "historical_app_ids": [r["app"] for r in rows],
        "historical_repositories": [r["repo"] for r in rows],
        "source_git_shas": [r["git_sha"] for r in rows],
        "source_html_shas": [r["html_sha256"] for r in rows],
    }
    attribution = {"schema": SCHEMA, "families": rows}
    return metadata, pixels, history, attribution


class SourceFamilyAttributionTests(unittest.TestCase):
    def test_valid_paired_metadata_never_approves_training_or_ood(self):
        result = audit_attribution(*_fixture())
        self.assertEqual(result["paired_source_families"], len(BANNED))
        self.assertEqual(result["frozen_h173_pairs_matched"], len(SOURCE_PINS))
        self.assertFalse(result["source_history_completeness_proven"])
        self.assertFalse(result["provenance_authenticity_proven"])
        self.assertFalse(result["model_training_approved"])
        self.assertFalse(result["ood_evaluation_approved"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["native_rewards"], 0)

    def test_pair_order_and_repo_case_do_not_change_fingerprint(self):
        args = _fixture()
        baseline = audit_attribution(*args)
        updated = deepcopy(args)
        updated[3]["families"].reverse()
        updated[3]["families"][0]["repo"] = updated[3]["families"][0]["repo"].upper()
        result = audit_attribution(*updated)
        self.assertEqual(result["attribution_sha256"], baseline["attribution_sha256"])

    def test_legacy_placeholder_repository_cannot_fake_family_coverage(self):
        meta, pixels, history, attribution = _fixture()
        history["historical_repositories"] = ["fictional-owner/placeholder"]
        with self.assertRaisesRegex(ValueError, "repository attribution"):
            audit_attribution(meta, pixels, history, attribution)

    def test_missing_known_source_family_refused(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"].pop()
        with self.assertRaisesRegex(ValueError, "missing known|disagree"):
            audit_attribution(meta, pixels, history, attribution)

    def test_known_h173_pair_must_not_be_substituted(self):
        for app in SOURCE_PINS:
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            row["git_sha"] = "e"*40
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "H173 frozen original app"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_historical_sha_must_be_attributed_to_a_family(self):
        meta, pixels, history, attribution = _fixture()
        history["source_git_shas"].append("d"*40)
        with self.assertRaisesRegex(ValueError, "Git SHA attribution"):
            audit_attribution(meta, pixels, history, attribution)

    def test_duplicate_historical_app_cannot_hide_alternate_revision(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"].append(deepcopy(attribution["families"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate historical application"):
            audit_attribution(meta, pixels, history, attribution)

    def test_case_variant_repo_alias_same_family(self):
        meta, pixels, history, attribution = _fixture()
        attribution["families"][0]["repo"] = "FICTIONAL-OWNER/OLD-ORIGINAL-1"
        result = audit_attribution(meta, pixels, history, attribution)
        self.assertEqual(result["distinct_historical_repositories"], len(BANNED))

    def test_no_images_gold_or_private_fields_in_attribution(self):
        for payload in ({"gold": "fake"}, {"image_sha256": "f"*64},
                        {"coordinates": [1, 2]}):
            meta, pixels, history, attribution = _fixture()
            attribution["families"][0].update(payload)
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "only app/repo/Git/HTML"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_type_confusion_and_malformed_repos_fail(self):
        for bad in (None, [], ["owner", "repo"], "owner/../repo", "owner/."):
            meta, pixels, history, attribution = _fixture()
            attribution["families"][0]["repo"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit_attribution(meta, pixels, history, attribution)

    def test_non_metadata_root_fields_rejected(self):
        meta, pixels, history, attribution = _fixture()
        attribution["raw_images"] = ["/private/path"]
        with self.assertRaisesRegex(ValueError, "exact metadata-only"):
            audit_attribution(meta, pixels, history, attribution)

    def test_known_historical_app_reuse_still_blocked(self):
        meta, pixels, history, attribution = _fixture()
        meta["sources"][0]["repo"] = attribution["families"][0]["repo"]
        with self.assertRaisesRegex(ValueError, "historical upstream"):
            audit_attribution(meta, pixels, history, attribution)

    def test_invalid_legacy_history_still_blocked(self):
        meta, pixels, history, attribution = _fixture()
        history["source_html_shas"].pop()
        with self.assertRaisesRegex(ValueError, "original HTML SHA attribution"):
            audit_attribution(meta, pixels, history, attribution)


if __name__ == "__main__":
    unittest.main()

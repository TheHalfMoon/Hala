"""H230: independent original public repo+SHA source metadata regression.

Every test is synthetic identity metadata only; no owner old screenshots,
gold, model outputs, OOD or training.
"""
from __future__ import annotations

from copy import deepcopy
import unittest

from h176.source_family_attribution import (
    FROZEN_OTHER_GIT, FROZEN_OTHER_REPOS,
    audit_attribution,
)
from h177.public_origin_pins import KNOWN_PUBLIC_PREVIOUS
from test_h176_source_family_attribution import _fixture

NEWLY_PINNED = {
    "vanilla-native-board": (
        "anantagarwal1307/drag-drop-kanban-board",
        "47cfba359a71b70a6a23e511e554416aec006474",
    ),
    "react-native-board": (
        "dandemslay/react-kanbanboard",
        "fa047f7a32afb57750ed5246456dc282ea7befcd",
    ),
    "simple-table": (
        "arce/simple-table",
        "7580270d18fb764bbf437a5e5b88ea66d63fc449",
    ),
    "calculator": (
        "ashishxa/calculator",
        "8f27a92072dde9658d8d94db8918a184ed7bd416",
    ),
}


class FourRemainingHistoricalOriginalSourceTests(unittest.TestCase):
    def test_all_nine_public_git_source_original_pairs_now_required_by_h176(self):
        self.assertEqual(set(FROZEN_OTHER_REPOS), set(KNOWN_PUBLIC_PREVIOUS))
        self.assertEqual(set(FROZEN_OTHER_REPOS), set(FROZEN_OTHER_GIT))
        self.assertEqual(len(FROZEN_OTHER_REPOS), 9)
        self.assertEqual(len(NEWLY_PINNED), 4)
        for app, (repo, sha) in KNOWN_PUBLIC_PREVIOUS.items():
            self.assertEqual(FROZEN_OTHER_REPOS[app], repo.casefold())
            self.assertEqual(FROZEN_OTHER_GIT[app], sha)

    def test_four_original_repo_substitution_fails_even_if_history_rewritten(self):
        for app, (_repo, sha) in NEWLY_PINNED.items():
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            original = row["repo"]
            fake = "forged-owner/fabricated-original-" + app
            row["repo"] = fake
            history["historical_repositories"].remove(original)
            history["historical_repositories"].append(fake)
            self.assertEqual(row["git_sha"], sha)
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "earlier source pilot/eval original upstream repository mismatch"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_four_original_sha_substitution_fails_even_if_history_rewritten(self):
        for app, (repo, _sha) in NEWLY_PINNED.items():
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            old = row["git_sha"]
            fake = "e" * 40
            row["git_sha"] = fake
            history["source_git_shas"].remove(old)
            history["source_git_shas"].append(fake)
            self.assertEqual(row["repo"].casefold(), repo)
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "earlier source pilot/eval frozen Git SHA mismatch"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_missing_old_source_in_both_owner_lists_still_fails(self):
        for app in NEWLY_PINNED:
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            attribution["families"].remove(row)
            for field, value in (
                ("historical_app_ids", row["app"]),
                ("historical_repositories", row["repo"]),
                ("source_git_shas", row["git_sha"]),
                ("source_html_shas", row["html_sha256"]),
            ):
                history[field].remove(value)
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "missing known historical TRAIN/DEV/OOD app or pinned pairing"
            ):
                audit_attribution(meta, pixels, history, attribution)

    def test_conventional_github_clone_case_suffix_preserves_same_source(self):
        for app, (repo, sha) in NEWLY_PINNED.items():
            meta, pixels, history, attribution = _fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            old = row["repo"]
            row["repo"] = repo.upper() + ".GIT"
            history["historical_repositories"].remove(old)
            history["historical_repositories"].append(row["repo"])
            self.assertEqual(row["git_sha"], sha)
            result = audit_attribution(meta, pixels, history, attribution)
            self.assertFalse(result["model_training_approved"])
            self.assertFalse(result["source_history_completeness_proven"])
            self.assertEqual(result["model_calls"], 0)

    def test_two_old_original_sources_swapped_without_roster_changes_rejected(self):
        apps = ("vanilla-native-board", "react-native-board")
        meta, pixels, history, attribution = _fixture()
        a = next(r for r in attribution["families"] if r["app"] == apps[0])
        b = next(r for r in attribution["families"] if r["app"] == apps[1])
        a["repo"], b["repo"] = b["repo"], a["repo"]
        with self.assertRaisesRegex(
            ValueError, "earlier source pilot/eval original upstream repository mismatch"
        ):
            audit_attribution(meta, pixels, history, attribution)

    def test_only_known_original_source_nine_does_not_prove_whole_history(self):
        result = audit_attribution(*_fixture())
        self.assertEqual(result["earlier_source_git_pins_matched"], 9)
        self.assertFalse(result["source_history_completeness_proven"])
        self.assertFalse(result["provenance_authenticity_proven"])
        self.assertFalse(result["model_training_approved"])
        self.assertFalse(result["ood_evaluation_approved"])
        self.assertEqual(result["native_rewards"], 0)


if __name__ == "__main__":
    unittest.main()

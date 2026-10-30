"""Only synthetic metadata, with PUBLIC H138-H164 immutable source pins.

Never open historic OOD media, private gold, inference traces or weights.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from h177.public_origin_pins import (
    KNOWN_PUBLIC_PREVIOUS, KNOWN_PUBLIC_REPOS, audit_public_origin_pins,
)
from test_h176_source_family_attribution import _fixture
from h177.audit_owner_roots import run as owner_cli


def fixture():
    metadata, old_pixels, history, attribution = deepcopy(_fixture())
    next_id = 1000
    by_app = {row["app"]: row for row in attribution["families"]}
    for app, (repo, sha) in KNOWN_PUBLIC_PREVIOUS.items():
        if app not in by_app:
            row = {
                "app": app, "repo": repo, "git_sha": sha,
                "html_sha256": f"{next_id:064x}",
            }
            next_id += 1
            attribution["families"].append(row)
            by_app[app] = row
        else:
            by_app[app]["repo"] = repo
            by_app[app]["git_sha"] = sha
    rows = attribution["families"]
    history["historical_app_ids"] = [r["app"] for r in rows]
    history["historical_repositories"] = [
        r["repo"] for r in rows if "repo" in r
    ]
    history["source_git_shas"] = [
        r["git_sha"] for r in rows if "git_sha" in r
    ]
    history["source_html_shas"] = [
        r["html_sha256"] for r in rows if "html_sha256" in r
    ]
    return metadata, old_pixels, history, attribution


class PublicHistoricalOriginTests(unittest.TestCase):
    def test_valid_synthetic_roster_not_dataset_or_training_approval(self):
        result = audit_public_origin_pins(*fixture())
        self.assertEqual(
            result["public_git_repository_pin_pairs_checked"],
            len(KNOWN_PUBLIC_PREVIOUS),
        )
        self.assertGreaterEqual(result["known_public_previous_app_ids_at_least"], 22)
        self.assertFalse(result["historical_source_completeness_proven"])
        self.assertFalse(result["old_html_source_authenticity_proven"])
        self.assertFalse(result["training_approved"])
        self.assertFalse(result["ood_evaluation_approved"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["native_rewards"], 0)

    def test_public_historical_git_pins_are_unique_and_well_formed(self):
        self.assertEqual(len(KNOWN_PUBLIC_PREVIOUS), 9)
        self.assertEqual(len(KNOWN_PUBLIC_REPOS), 9)
        self.assertEqual(
            len({sha for _repo, sha in KNOWN_PUBLIC_PREVIOUS.values()}), 9
        )
        for _app, (repo, sha) in KNOWN_PUBLIC_PREVIOUS.items():
            self.assertEqual(len(sha), 40)
            self.assertEqual(len(repo.split("/")), 2)

    def test_forged_repo_cannot_hide_true_known_original_upstream(self):
        for app in KNOWN_PUBLIC_PREVIOUS:
            metadata, pixel, history, attribution = fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            before = row["repo"]
            row["repo"] = "forger/source"
            history["historical_repositories"].remove(before)
            history["historical_repositories"].append(row["repo"])
            # All nine known public Git originals are rejected by H176
            # before the H177 independently pinned broader source check.
            # Both pathways must reject the forgery; neither is admission.
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError,
                "source-repository or Git pin contradicts|"
                "earlier source pilot/eval original upstream repository mismatch",
            ):
                audit_public_origin_pins(
                    metadata, pixel, history, attribution
                )

    def test_older_training_board_missing_even_if_h176_minimum_is_met(self):
        for app in (
            "vanilla-native-board", "react-native-board", "simple-table"
        ):
            metadata, pixel, history, attribution = fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            attribution["families"].remove(row)
            for key, val in (
                ("historical_app_ids", row["app"]),
                ("historical_repositories", row["repo"]),
                ("source_git_shas", row["git_sha"]),
                ("source_html_shas", row["html_sha256"]),
            ):
                history[key].remove(val)
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError,
                "known prior TRAIN/DEV/OOD source attribution missing|"
                "missing known historical TRAIN/DEV/OOD app or pinned pairing",
            ):
                audit_public_origin_pins(
                    metadata, pixel, history, attribution
                )

    def test_mismatched_old_git_sha_rejected(self):
        # All immutable public source SHA pins must remain exact. For five
        # known H176 pins, H176 itself may reject before H177 does.
        for app in KNOWN_PUBLIC_PREVIOUS:
            metadata, pixel, history, attribution = fixture()
            row = next(r for r in attribution["families"] if r["app"] == app)
            before = row["git_sha"]
            row["git_sha"] = "e"*40
            history["source_git_shas"].remove(before)
            history["source_git_shas"].append(row["git_sha"])
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "Git|pin"
            ):
                audit_public_origin_pins(
                    metadata, pixel, history, attribution
                )

    def test_different_case_of_known_original_repositories_is_accepted(self):
        metadata, pixel, history, attribution = fixture()
        app = "vanilla-native-board"
        row = next(r for r in attribution["families"] if r["app"] == app)
        before = row["repo"]
        row["repo"] = before.swapcase()
        history["historical_repositories"].remove(before)
        history["historical_repositories"].append(row["repo"])
        result = audit_public_origin_pins(
            metadata, pixel, history, attribution
        )
        self.assertEqual(result["public_git_repository_pin_pairs_checked"], 9)

    def test_fresh_candidate_cannot_reuse_known_historical_repository(self):
        for app, (repo, _) in KNOWN_PUBLIC_PREVIOUS.items():
            metadata, pixel, history, attribution = fixture()
            metadata["sources"][0]["repo"] = repo.upper()
            with self.subTest(app=app), self.assertRaisesRegex(
                ValueError, "historical upstream repository"
            ):
                audit_public_origin_pins(
                    metadata, pixel, history, attribution
                )

    def test_owner_cli_returns_only_safe_counts_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [
                Path(root) / name for name in (
                    "candidate.json", "prior_hash.json",
                    "historical.json", "paired.json"
                )
            ]
            for path, payload in zip(paths, fixture()):
                path.write_text(json.dumps(payload), encoding="utf-8")
            args = [
                "--candidates", str(paths[0]),
                "--prior-pixel-hashes", str(paths[1]),
                "--prior-source-history", str(paths[2]),
                "--paired-source-history", str(paths[3]),
            ]
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = owner_cli(args)
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertIn("HISTORICAL_COMPLETENESS_UNPROVEN", out.getvalue())
            self.assertNotIn(str(Path(root)), out.getvalue())
            self.assertNotIn("fictional-owner", out.getvalue())
            self.assertNotIn("arce/simple-table", out.getvalue())

    def test_owner_cli_reports_failure_without_private_paths(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [
                Path(root) / name for name in (
                    "candidate.json", "prior_hash.json",
                    "historical.json", "paired.json"
                )
            ]
            inputs = list(fixture())
            false_attribution = deepcopy(inputs[3])
            row = next(r for r in false_attribution["families"]
                       if r["app"] == "simple-table")
            original = row["repo"]
            row["repo"] = "attacker/original-owner-source"
            inputs[2]["historical_repositories"].remove(original)
            inputs[2]["historical_repositories"].append(row["repo"])
            inputs[3] = false_attribution
            for path, payload in zip(paths, inputs):
                path.write_text(json.dumps(payload), encoding="utf-8")
            args = [
                "--candidates", str(paths[0]),
                "--prior-pixel-hashes", str(paths[1]),
                "--prior-source-history", str(paths[2]),
                "--paired-source-history", str(paths[3]),
            ]
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = owner_cli(args)
            self.assertEqual(code, 2)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("H177_FAIL_CLOSED", err.getvalue())
            self.assertNotIn("attacker", err.getvalue())
            self.assertNotIn(str(Path(root)), err.getvalue())

    def test_h176_source_disjoint_guard_still_applies(self):
        metadata, pixel, history, attribution = fixture()
        metadata["sources"][1]["repo"] = metadata["sources"][0]["repo"]
        with self.assertRaisesRegex(
            ValueError, "duplicate upstream repository"
        ):
            audit_public_origin_pins(metadata, pixel, history, attribution)

    def test_legacy_source_history_missing_ood_pin_fails_closed(self):
        metadata, pixel, history, attribution = fixture()
        history["source_git_shas"].remove(
            "9eadca70c72ef9adb9144cc23a4104978d3bc63b"
        )
        with self.assertRaises(ValueError):
            audit_public_origin_pins(
                metadata, pixel, history, attribution
            )


if __name__ == "__main__":
    unittest.main()

"""H179 synthetic source-only donor novelty tests, no media or GPU."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h173.source_preflight import SOURCE_PINS
from h174.source_candidates import CANDIDATES
from h177.public_origin_pins import KNOWN_PUBLIC_PREVIOUS
from h179.audit_pinned_candidates import run as cli_run
from h179.source_novelty import (
    screen_h174_candidates, screen_pins_against_history,
)
from test_h178_owner_git_inventory import _synthetic_original


class OriginalSourceNoveltyTests(unittest.TestCase):
    def test_fixed_candidate_identities_disjoint_from_old_local_git_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            proof = screen_h174_candidates([folder])
            self.assertEqual(proof["candidate_source_families"], 3)
            self.assertEqual(proof["historical_git_checkouts_screened"], 1)
            self.assertEqual(proof["independent_local_source_byte_pins_verified"], 0)
            self.assertFalse(proof["train_dev_ood_assigned"])
            self.assertFalse(proof["source_history_completeness_proven"])
            self.assertFalse(proof["training_approved"])
            self.assertFalse(proof["ood_evaluation_approved"])
            self.assertEqual(proof["model_calls"], 0)

    def test_local_used_original_git_repository_rejected_even_other_revision(self):
        for app in CANDIDATES:
            with self.subTest(app=app), tempfile.TemporaryDirectory() as folder:
                _synthetic_original(Path(folder) / "private-h139")
                pins = deepcopy(CANDIDATES)
                pins[app]["repo"] = "MOCK-OWNER/USED-BOARD"
                with self.assertRaisesRegex(ValueError, "previously observed"):
                    screen_pins_against_history(pins, [folder])

    def test_new_git_sha_aliases_existing_old_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            _root, sha = _synthetic_original(Path(folder) / "private-h139")
            pins = deepcopy(CANDIDATES)
            pins["pickr"]["git_sha"] = sha
            with self.assertRaisesRegex(ValueError, "historically reused"):
                screen_pins_against_history(pins, [folder])

    def test_public_original_upstreams_rejected_without_owner_checkout(self):
        for app, (repo, _) in KNOWN_PUBLIC_PREVIOUS.items():
            with self.subTest(app=app), tempfile.TemporaryDirectory() as folder:
                _synthetic_original(Path(folder) / "private-h139")
                pins = deepcopy(CANDIDATES)
                pins["pickr"]["repo"] = repo.upper()
                with self.assertRaisesRegex(ValueError, "previously observed"):
                    screen_pins_against_history(pins, [folder])

    def test_h173_frozen_git_or_original_html_cannot_be_reused(self):
        for app, frozen in SOURCE_PINS.items():
            for field, value in (
                ("git_sha", frozen["git_sha"]),
                ("html_sha256", frozen["original_html_sha256"]),
            ):
                with self.subTest(app=app, field=field), tempfile.TemporaryDirectory() as folder:
                    _synthetic_original(Path(folder) / "private-h139")
                    pins = deepcopy(CANDIDATES)
                    pins["pickr"][field] = value
                    with self.assertRaisesRegex(ValueError, "historically reused"):
                        screen_pins_against_history(pins, [folder])

    def test_h173_original_repo_and_npm_upstream_cannot_reenter(self):
        for repo in ("robinparisi/tingle", "ganlanyuan/tiny-slider",
                     "nathancahill/split"):
            with self.subTest(repo=repo), tempfile.TemporaryDirectory() as folder:
                _synthetic_original(Path(folder) / "private-h139")
                pins = deepcopy(CANDIDATES)
                pins["pickr"]["repo"] = repo.upper()
                with self.assertRaisesRegex(ValueError, "previously observed"):
                    screen_pins_against_history(pins, [folder])

    def test_repeated_donor_repository_git_or_html_rejected(self):
        for key in ("repo", "git_sha", "html_sha256"):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as folder:
                _synthetic_original(Path(folder) / "private-h139")
                pins = deepcopy(CANDIDATES)
                pins["medium-zoom"][key] = pins["pickr"][key]
                with self.assertRaisesRegex(ValueError, "duplicate candidate"):
                    screen_pins_against_history(pins, [folder])

    def test_forbidden_historical_app_or_extra_private_fields_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            pins = deepcopy(CANDIDATES)
            pins["choices"] = pins.pop("pickr")
            with self.assertRaisesRegex(ValueError, "previously used"):
                screen_pins_against_history(pins, [folder])
            pins = deepcopy(CANDIDATES)
            pins["pickr"]["owner_gold"] = "private"
            with self.assertRaisesRegex(ValueError, "only exact pinned"):
                screen_pins_against_history(pins, [folder])

    def test_bad_source_path_or_license_sha_is_rejected(self):
        for field, value in (
            ("html_path", "../escape.html"),
            ("license_path", "/private/secrets"),
            ("license_sha256", "bad"),
            ("license", "Unknown"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                _synthetic_original(Path(folder) / "private-h139")
                pins = deepcopy(CANDIDATES)
                pins["pickr"][field] = value
                with self.assertRaises(ValueError):
                    screen_pins_against_history(pins, [folder])

    def test_empty_or_missing_owner_history_must_fail_closed(self):
        for arg in ([], ["/missing-private-source-root"]):
            with self.subTest(arg=arg), self.assertRaises(ValueError):
                screen_h174_candidates(arg)

    def test_all_three_original_source_byte_attestations_required(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            with self.assertRaisesRegex(ValueError, "all three original"):
                screen_h174_candidates([folder], {"pickr": "/incomplete"})
            expected = {
                name: "/not-used-test/" + name for name in CANDIDATES
            }
            def verify(name, _path):
                return {
                    "source_git_sha": CANDIDATES[name]["git_sha"],
                    "original_html_sha256": CANDIDATES[name]["html_sha256"],
                }
            with patch("h179.source_novelty.verify_local_source_only",
                       side_effect=verify) as fake:
                proof = screen_h174_candidates([folder], expected)
            self.assertEqual(fake.call_count, len(CANDIDATES))
            self.assertEqual(
                proof["independent_local_source_byte_pins_verified"], 3,
            )
            self.assertFalse(proof["training_approved"])

    def test_original_source_byte_forgery_aborts_all_attestation(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            donors = {name: "test/" + name for name in CANDIDATES}
            with patch(
                "h179.source_novelty.verify_local_source_only",
                return_value={"source_git_sha": "0"*40,
                              "original_html_sha256": "1"*64},
            ):
                with self.assertRaisesRegex(ValueError, "byte identity mismatch"):
                    screen_h174_candidates([folder], donors)

    def test_local_cli_omits_owner_source_paths_and_names(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = cli_run(["--historical-root", folder])
            self.assertEqual(exit_code, 0)
            self.assertIn("candidates=3", out.getvalue())
            self.assertIn("old_checkouts=1", out.getvalue())
            self.assertIn("HISTORICAL_COMPLETENESS_UNPROVEN", out.getvalue())
            self.assertNotIn(folder, out.getvalue())
            self.assertNotIn("mock-owner", out.getvalue())
            self.assertEqual(err.getvalue(), "")

    def test_local_cli_partial_donor_set_fail_closed_no_private_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            _synthetic_original(Path(folder) / "private-h139")
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = cli_run(
                    ["--historical-root", folder, "--pickr-donor", folder]
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("H179_FAIL_CLOSED_INCOMPLETE", err.getvalue())
            self.assertNotIn(folder, err.getvalue())


if __name__ == "__main__":
    unittest.main()

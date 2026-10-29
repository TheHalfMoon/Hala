"""H181 synthetic source reservation tests. Never read an OOD image."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from h174.source_candidates import CANDIDATES
from h181.audit_owner_reservation import run as cli_run
from h181.ood_source_reservation import (
    GANTT_PIN, SCHEMA, SOURCE_ROLE, check_reserved_source_only,
    preregistered_identity,
)
from test_h178_owner_git_inventory import _synthetic_original


def source_root(root: str, origin: str = "https://github.com/mock-owner/old-source.git"):
    phase = Path(root) / "private-h140"
    _synthetic_original(phase, origin=origin)


class GanttSourceReservationTests(unittest.TestCase):
    def test_exact_immutable_gantt_identity_pins(self):
        self.assertEqual(GANTT_PIN["repo"], "frappe/gantt")
        self.assertEqual(len(GANTT_PIN["git_sha"]), 40)
        self.assertEqual(len(GANTT_PIN["html_sha256"]), 64)
        self.assertEqual(len(GANTT_PIN["license_sha256"]), 64)
        self.assertEqual(GANTT_PIN["license"], "MIT")
        self.assertEqual(preregistered_identity()["schema"], SCHEMA)
        self.assertEqual(preregistered_identity()["role"], SOURCE_ROLE)

    def test_valid_synthetic_old_git_inventory_only_reserves_source(self):
        with tempfile.TemporaryDirectory() as root:
            source_root(root)
            result = check_reserved_source_only(preregistered_identity(), [root])
            self.assertEqual(result["owner_historical_git_checkouts_screened"], 1)
            self.assertEqual(result["previous_original_app_sources_in_this_branch"], 3)
            self.assertFalse(result["source_history_completeness_proven"])
            self.assertFalse(result["source_fork_equivalence_ruled_out"])
            self.assertFalse(result["training_data_admitted"])
            self.assertFalse(result["dev_data_admitted"])
            self.assertFalse(result["ood_evaluation_approved"])
            self.assertEqual(result["ood_images_created"], 0)
            self.assertEqual(result["ood_model_inputs_created"], 0)
            self.assertEqual(result["model_calls"], 0)

    def test_original_git_sha_html_license_fields_are_immutable(self):
        for key, val in (
            ("repo", "attacker/hidden-fork"),
            ("git_sha", "0"*40),
            ("html_path", "other-demo.html"),
            ("html_sha256", "f"*64),
            ("license_path", "other-license.txt"),
            ("license_sha256", "f"*64),
            ("license", "Unknown"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                source_root(root)
                row = deepcopy(preregistered_identity())
                row["source"][key] = val
                with self.assertRaisesRegex(ValueError, "immutable original"):
                    check_reserved_source_only(row, [root])

    def test_unseen_ood_rejects_any_observed_git_repo_case_alias(self):
        with tempfile.TemporaryDirectory() as root:
            source_root(root, origin="https://github.com/FRAPPE/GANTT.git")
            with self.assertRaisesRegex(ValueError, "previously observed"):
                check_reserved_source_only(preregistered_identity(), [root])

    def test_previously_exposed_three_candidates_not_sealed_ood(self):
        self.assertEqual(len(CANDIDATES), 3)
        self.assertNotIn(
            GANTT_PIN["repo"].casefold(),
            {row["repo"].casefold() for row in CANDIDATES.values()},
        )

    def test_ood_role_cannot_be_advanced_to_capture(self):
        for role in ("OOD_READY", "TRAIN", "DEV", "OOD_EVALUATED"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                source_root(root)
                row = deepcopy(preregistered_identity())
                row["role"] = role
                with self.assertRaisesRegex(ValueError, "exact OOD source-only"):
                    check_reserved_source_only(row, [root])

    def test_no_gold_or_owner_task_fields_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            source_root(root)
            row = deepcopy(preregistered_identity())
            row["gold_coordinate"] = [10, 20]
            with self.assertRaisesRegex(ValueError, "exact OOD source-only"):
                check_reserved_source_only(row, [root])
            row = deepcopy(preregistered_identity())
            row["source"]["screenshot"] = "ood-private"
            with self.assertRaises(ValueError):
                check_reserved_source_only(row, [root])

    def test_missing_owner_source_census_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "private-h140").mkdir()
            with self.assertRaisesRegex(ValueError, "no verified original"):
                check_reserved_source_only(preregistered_identity(), [root])

    def test_source_fingerprint_repeatable_and_source_only(self):
        with tempfile.TemporaryDirectory() as root:
            source_root(root)
            a = check_reserved_source_only(preregistered_identity(), [root])
            b = check_reserved_source_only(preregistered_identity(), [root])
            self.assertEqual(
                a["source_reservation_fingerprint_sha256"],
                b["source_reservation_fingerprint_sha256"],
            )
            self.assertEqual(len(a["source_reservation_fingerprint_sha256"]), 64)

    def test_cli_prints_counts_not_private_repo_or_root(self):
        with tempfile.TemporaryDirectory() as root:
            source_root(root)
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = cli_run(["--historical-root", root])
            self.assertEqual(exit_code, 0)
            self.assertIn("NO_OOD_IMAGES_NO_MODEL_CALLS", out.getvalue())
            self.assertIn("old_repositories=1", out.getvalue())
            self.assertNotIn(root, out.getvalue())
            self.assertNotIn("mock-owner", out.getvalue())
            self.assertEqual(err.getvalue(), "")

    def test_cli_missing_root_failure_never_echoes_owner_path(self):
        with tempfile.TemporaryDirectory() as root:
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = cli_run(["--historical-root", root])
            self.assertEqual(exit_code, 2)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("H181_FAIL_CLOSED", err.getvalue())
            self.assertNotIn(root, err.getvalue())


if __name__ == "__main__":
    unittest.main()

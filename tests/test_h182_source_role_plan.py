"""H182 prospective TRAIN/DEV original source roles: synthetic-only tests.

All fake old Git repository fixtures are disposable and use dummy commits.
Never open consumed H173 pixels, owner gold, screenshots or model weights.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from h174.source_candidates import CANDIDATES
from h181.ood_source_reservation import GANTT_PIN, SOURCE_ROLE
from h182.audit_owner_split_plan import run as owner_cli
from h182.source_role_plan import (
    SCHEMA, SPEC, audit_preregistered_sources, preregistered_plan,
)
from test_h178_owner_git_inventory import _synthetic_original


def _fixture():
    root = tempfile.TemporaryDirectory()
    _synthetic_original(Path(root.name) / "private-h140")
    return root


class OriginalSourceProspectiveRoleTests(unittest.TestCase):
    def test_preregistered_roles_precede_all_captures(self):
        plan = preregistered_plan()
        self.assertEqual(plan["schema"], SCHEMA)
        self.assertEqual(len(plan["roles"]), 4)
        self.assertEqual(
            [(row["app"], row["role"]) for row in plan["roles"]],
            list(SPEC),
        )
        self.assertEqual(plan["roles"][-1]["role"], SOURCE_ROLE)
        self.assertEqual(plan["roles"][-1]["source"], GANTT_PIN)

    def test_valid_synthetic_local_history_is_never_training_approval(self):
        with _fixture() as root:
            proof = audit_preregistered_sources(preregistered_plan(), [root])
            self.assertEqual(proof["prospective_train_source_count"], 1)
            self.assertEqual(proof["prospective_dev_source_count"], 1)
            self.assertEqual(proof["exposed_research_unassigned_source_count"], 1)
            self.assertEqual(proof["tentatively_reserved_ood_source_count"], 1)
            self.assertEqual(proof["pinned_pristine_original_donor_files_verified"], 0)
            self.assertFalse(proof["complete_history_proven"])
            self.assertFalse(proof["historical_pixel_sha_coverage_proven"])
            self.assertFalse(proof["native_train_dev_dataset_qualified"])
            self.assertFalse(proof["training_approved"])
            self.assertFalse(proof["ood_evaluation_approved"])
            self.assertEqual(proof["train_images_created"], 0)
            self.assertEqual(proof["dev_images_created"], 0)
            self.assertEqual(proof["ood_images_created"], 0)
            self.assertEqual(proof["model_calls"], 0)

    def test_all_roles_exact_and_immutable(self):
        for row_id in range(4):
            for key, bad in (
                ("role", "TRAIN"),
                ("app", "other-app"),
            ):
                with self.subTest(row=row_id, field=key):
                    plan = deepcopy(preregistered_plan())
                    plan["roles"][row_id][key] = bad
                    with self.assertRaises(ValueError):
                        audit_preregistered_sources(plan, [])

    def test_no_swapping_train_dev_families(self):
        plan = preregistered_plan()
        plan["roles"][0], plan["roles"][1] = (
            plan["roles"][1], plan["roles"][0]
        )
        with self.assertRaisesRegex(ValueError, "immutable original"):
            audit_preregistered_sources(plan, [])

    def test_no_gantt_ood_fingerprint_mutation(self):
        for field, bad in (
            ("repo", "attacker/ood-candidate"),
            ("git_sha", "a"*40),
            ("html_sha256", "f"*64),
            ("license", "OTHER"),
        ):
            with self.subTest(field=field):
                plan = preregistered_plan()
                plan["roles"][-1]["source"][field] = bad
                with self.assertRaisesRegex(ValueError, "immutable original"):
                    audit_preregistered_sources(plan, [])

    def test_no_model_or_gold_or_image_fields(self):
        for owner_key in (
            "image_path", "reference_gold", "ood_screenshot",
            "model_weights", "action_xy",
        ):
            plan = preregistered_plan()
            plan["roles"][0][owner_key] = "sensitive"
            with self.subTest(owner_key=owner_key), self.assertRaisesRegex(
                ValueError, "role metadata only"
            ):
                audit_preregistered_sources(plan, [])
        plan = preregistered_plan()
        plan["cases"] = [{"secret": "screenshot"}]
        with self.assertRaisesRegex(ValueError, "unknown top-level"):
            audit_preregistered_sources(plan, [])

    def test_no_screenshot_ood_split_or_train_promotion(self):
        for role in ("OOD", "OOD_QUALIFIED", "TRAIN", "DEV"):
            plan = preregistered_plan()
            plan["roles"][3]["role"] = role
            with self.subTest(role=role), self.assertRaises(ValueError):
                audit_preregistered_sources(plan, [])

    def test_no_duplicate_repos_between_sources(self):
        plan = preregistered_plan()
        plan["roles"][1]["source"] = deepcopy(plan["roles"][0]["source"])
        with self.assertRaises(ValueError):
            audit_preregistered_sources(plan, [])

    def test_actual_source_pins_must_remain_exact(self):
        for app in CANDIDATES:
            plan = preregistered_plan()
            target = next(row for row in plan["roles"] if row["app"] == app)
            target["source"]["git_sha"] = "0"*40
            with self.subTest(app=app), self.assertRaises(ValueError):
                audit_preregistered_sources(plan, [])

    def test_historical_missing_root_is_not_approved(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "historical phase"):
                audit_preregistered_sources(preregistered_plan(), [root])

    def test_owner_cli_private_paths_never_displayed(self):
        with _fixture() as root:
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = owner_cli(["--historical-root", root])
            self.assertEqual(code, 0)
            self.assertIn("HISTORY_INCOMPLETE_NO_CASE_DATA", stdout.getvalue())
            self.assertIn("train_sources=1", stdout.getvalue())
            self.assertIn("ood_reserved_sources=1", stdout.getvalue())
            self.assertNotIn(root, stdout.getvalue())
            self.assertNotIn("mock-owner", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_owner_cli_partial_donors_no_sensitive_path_leak(self):
        with _fixture() as root:
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = owner_cli([
                    "--historical-root", root,
                    "--pickr-donor", root,
                ])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("H182_FAIL_CLOSED_PARTIAL", stderr.getvalue())
            self.assertNotIn(root, stderr.getvalue())

    def test_pristine_local_donors_all_three_mandatory(self):
        with _fixture() as root:
            with self.assertRaisesRegex(ValueError, "all three original"):
                audit_preregistered_sources(
                    preregistered_plan(), [root], {"pickr": root}
                )

    def test_donor_original_byte_mismatch_rejected(self):
        with _fixture() as root:
            bad = {app: root for app in CANDIDATES}
            with patch("h179.source_novelty.verify_local_source_only",
                       side_effect=ValueError("source bytes failed")):
                with self.assertRaisesRegex(ValueError, "source bytes failed"):
                    audit_preregistered_sources(
                        preregistered_plan(), [root], bad
                    )


if __name__ == "__main__":
    unittest.main()

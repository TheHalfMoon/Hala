"""H184 source-only prior-public-origin non-recycling tests.

All Git roots are synthetic tempfile repositories. Historical owner source
content, training images, heldout labels, model weights remain untouched.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h174.source_candidates import CANDIDATES
from h177.public_origin_pins import KNOWN_PUBLIC_REPOS
from h179.source_novelty import (
    screen_h174_candidates, screen_pins_against_history,
)
from h181.ood_source_reservation import (
    check_reserved_source_only, preregistered_identity,
)
from h184.public_prior_considered_origins import (
    BLOCKED_REPOSITORIES, BLOCKED_COMMITS,
    PUBLIC_PREVIOUS_OR_CONSIDERED, validate_public_previous_ledger,
)
from test_h178_owner_git_inventory import _synthetic_original


class PublicHistoricalOriginsNoRecyclingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _synthetic_original(Path(cls.tmp.name) / "private-h140")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ledger_shape_status_and_no_completeness_promise(self):
        proof = validate_public_previous_ledger()
        self.assertEqual(proof["public_extra_repositories"], 9)
        self.assertEqual(proof["public_extra_git_pins"], 9)
        self.assertFalse(proof["complete_history_proven"])
        self.assertFalse(proof["training_approved"])
        self.assertEqual(proof["model_calls"], 0)

    def test_public_h139_consumed_two_train_and_one_blind_locked(self):
        rows = PUBLIC_PREVIOUS_OR_CONSIDERED
        self.assertEqual(
            sum(row[3] == "past-train-dev" for row in rows.values()), 2
        )
        self.assertEqual(
            sum(row[3] == "past-blind" for row in rows.values()), 1
        )

    def test_public_proposed_rejected_and_source_qa_not_mislabeled_consumed(self):
        counts = {cls: sum(row[3] == cls for row
                   in PUBLIC_PREVIOUS_OR_CONSIDERED.values())
                  for cls in ("source-pilot", "rejected", "proposed")}
        self.assertEqual(counts["source-pilot"], 3)
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["proposed"], 2)

    def test_nine_upstreams_not_subsets_of_older_nine(self):
        self.assertEqual(len(BLOCKED_REPOSITORIES), 9)
        self.assertTrue(BLOCKED_REPOSITORIES.isdisjoint(KNOWN_PUBLIC_REPOS))

    def test_existing_three_candidate_sourcing_still_allowed_for_research_only(self):
        proof = screen_h174_candidates([self.tmp.name])
        self.assertEqual(proof["extra_public_historic_source_families_blocked"], 9)
        self.assertFalse(proof["training_approved"])
        self.assertFalse(proof["source_history_completeness_proven"])
        self.assertEqual(proof["independent_local_source_byte_pins_verified"], 0)

    def test_all_nine_prior_public_repos_blocked_regardless_of_git_revision(self):
        for name, (repo, _sha, _evidence, _kind) in (
            PUBLIC_PREVIOUS_OR_CONSIDERED.items()
        ):
            with self.subTest(app=name):
                pins = deepcopy(CANDIDATES)
                pins["pickr"]["repo"] = repo.upper()
                with self.assertRaisesRegex(ValueError, "previously observed"):
                    screen_pins_against_history(pins, [self.tmp.name])

    def test_all_nine_prior_exact_git_ids_blocked_under_brand_new_repo_name(self):
        for name, (_repo, sha, _evidence, _kind) in (
            PUBLIC_PREVIOUS_OR_CONSIDERED.items()
        ):
            with self.subTest(app=name):
                pins = deepcopy(CANDIDATES)
                pins["pickr"]["repo"] = "somebrand/new-source"
                pins["pickr"]["git_sha"] = sha
                with self.assertRaisesRegex(ValueError, "historically reused"):
                    screen_pins_against_history(pins, [self.tmp.name])

    def test_ood_gantt_stays_reserved_source_only_on_same_historic_gate(self):
        proof = check_reserved_source_only(
            preregistered_identity(), [self.tmp.name]
        )
        self.assertFalse(proof["ood_evaluation_approved"])
        self.assertFalse(proof["source_history_completeness_proven"])
        self.assertEqual(proof["ood_images_created"], 0)
        self.assertEqual(proof["model_calls"], 0)

    def test_fabricated_sha_or_repository_in_public_prior_ledger_fails_closed(self):
        for field, value in ((0, "badrepo"), (1, "BAD_SHA"),
                             (2, "private/GOLD.txt"), (3, "evaluation-pass")):
            with self.subTest(field=field):
                mutable = dict(PUBLIC_PREVIOUS_OR_CONSIDERED)
                first = next(iter(mutable))
                row = list(mutable[first])
                row[field] = value
                mutable[first] = tuple(row)
                with patch(
                    "h184.public_prior_considered_origins.PUBLIC_PREVIOUS_OR_CONSIDERED",
                    mutable,
                ), self.assertRaises(ValueError):
                    screen_pins_against_history(CANDIDATES, [self.tmp.name])

    def test_duplicate_public_prior_source_must_fail(self):
        mutable = dict(PUBLIC_PREVIOUS_OR_CONSIDERED)
        keys = list(mutable)
        mutable[keys[1]] = mutable[keys[0]]
        with patch(
            "h184.public_prior_considered_origins.PUBLIC_PREVIOUS_OR_CONSIDERED",
            mutable,
        ), self.assertRaisesRegex(ValueError, "duplicate"):
            validate_public_previous_ledger()

    def test_original_public_ledger_contains_no_owner_gold_or_images(self):
        for name, (repo, sha, path, cls) in (
            PUBLIC_PREVIOUS_OR_CONSIDERED.items()
        ):
            self.assertEqual(len(sha), 40)
            self.assertIn(path.split("/")[0], {"h139", "h140", "h154"})
            self.assertNotIn("private-", repo)
            self.assertNotIn("gold", path.casefold())
            self.assertNotIn("image", path.casefold())
            self.assertNotIn("pred", path.casefold())


if __name__ == "__main__":
    unittest.main()

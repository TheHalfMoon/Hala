"""H238 public historical source union and ENTRYPOINT regression.

All identities here are already PUBLIC project metadata. No historical
private files, owner images, weights, gold or consumed OOD accessed.
"""
import unittest
from copy import deepcopy
from unittest.mock import patch

from h174.source_candidates import CANDIDATES
from h179.source_novelty import screen_pins_against_history
from h174.split_firewall import HISTORY_SCHEMA, SCHEMA, validate
from h176.source_family_attribution import FROZEN_OTHER_REPOS, KNOWN_NPM_UPSTREAM_REPOS
from h177.public_origin_pins import KNOWN_PUBLIC_REPOS
from h184.public_prior_considered_origins import BLOCKED_REPOSITORIES
from h238.historical_family_floor import (
    known_public_source_family_floor, public_source_floor_receipt,
)


def synthetic_fresh_metadata(repo: str, split: str = "TRAIN") -> tuple[dict, dict]:
    sources = [
        {"app": "freshtrain", "split": "TRAIN", "repo": "synthetic/train",
         "git_sha": "1"*40, "html_sha256": "1"*64, "license": "MIT"},
        {"app": "freshdev", "split": "DEV", "repo": "synthetic/dev",
         "git_sha": "2"*40, "html_sha256": "2"*64, "license": "MIT"},
        {"app": "freshhold", "split": "OOD_RESERVED", "repo": "synthetic/hold",
         "git_sha": "3"*40, "html_sha256": "3"*64, "license": "MIT"},
    ]
    index = {"TRAIN": 0, "DEV": 1, "OOD_RESERVED": 2}[split]
    sources[index]["repo"] = repo
    cases = [
        {"index": 1, "app": "freshtrain", "split": "TRAIN",
         "image_sha256": "4"*64, "visual_rgba_sha256": "5"*64,
         "language": "en", "action_type": "click",
         "instruction": "Click this synthetic visible button"},
        {"index": 2, "app": "freshdev", "split": "DEV",
         "image_sha256": "6"*64, "visual_rgba_sha256": "7"*64,
         "language": "ar", "action_type": "drag",
         "instruction": "اسحب العنصر إلى الهدف"},
    ]
    registry = {
        "schema": HISTORY_SCHEMA,
        "image_shas": ["8"*64], "visual_shas": ["9"*64],
    }
    return {"schema": SCHEMA, "sources": sources, "cases": cases}, registry


class H238PublicHistoricalFamilyFloor(unittest.TestCase):
    def test_public_ledgers_union_has_27_unique_source_families(self):
        floor = known_public_source_family_floor()
        self.assertEqual(len(floor), 27)
        self.assertTrue(set(FROZEN_OTHER_REPOS.values()).issubset(floor))
        self.assertTrue(KNOWN_NPM_UPSTREAM_REPOS.issubset(floor))
        self.assertTrue(KNOWN_PUBLIC_REPOS.issubset(floor))
        self.assertTrue(BLOCKED_REPOSITORIES.issubset(floor))
        for name in ("choices-js/choices", "flatpickr/flatpickr",
                     "leaverou/awesomplete", "nudeui/awesomplete",
                     "robinparisi/tingle", "bevacqua/dragula"):
            self.assertEqual(name in floor, name != "bevacqua/dragula")

    def test_all_27_older_families_blocked_in_every_split(self):
        for repo in known_public_source_family_floor():
            for split in ("TRAIN", "DEV", "OOD_RESERVED"):
                meta, prior = synthetic_fresh_metadata(repo, split)
                with self.subTest(repo=repo, split=split), self.assertRaisesRegex(
                        ValueError,
                        "historical original source family|previous original TRAIN/DEV"):
                    validate(meta, prior)

    def test_clone_suffix_and_capitalization_cannot_evade_old_family(self):
        for repo in known_public_source_family_floor():
            for variant in (repo.upper(), repo+".git",
                            repo.upper()+".GIT", repo+".git.git"):
                meta, prior = synthetic_fresh_metadata(variant)
                with self.subTest(repo=variant), self.assertRaisesRegex(
                        ValueError,
                        "historical original source family|previous original TRAIN/DEV"):
                    validate(meta, prior)

    def test_public_previous_source_candidates_not_fresh(self):
        for repo in ("codewithsindhu/kanban-board-js",
                     "farama-foundation/miniwob-plusplus",
                     "n3xt3r1337/kanban-board",
                     "arjundroid12/todo-drag-drop"):
            self.assertIn(repo, known_public_source_family_floor())

    def test_genuinely_distinct_synthetic_source_still_metadata_preflight_only(self):
        meta, registry = synthetic_fresh_metadata("synthetic/source-unique-238")
        result = validate(meta, registry)
        self.assertEqual(result["train_cases"], 1)
        self.assertEqual(result["dev_cases"], 1)
        self.assertEqual(result["native_rewards"], 0)

    def test_h179_candidate_screen_blocks_old_repo_even_with_git_clone_suffix(self):
        # Stub ONLY a source-identity inventory result, never owner media.
        synthetic_inventory = {
            "_owner_only_repository_shas": set(),
            "source_checkouts_inspected": 1,
            "distinct_repositories": 0,
            "historical_identity_fingerprint_sha256": "0" * 64,
        }
        with patch("h179.source_novelty.inventory_local_original_sources",
                   return_value=synthetic_inventory):
            for repo in ("choices-js/choices.git", "nudeui/awesomplete.git.git",
                         "codewithsindhu/kanban-board-js.GIT",
                         "farama-foundation/miniwob-plusplus.git"):
                pins = deepcopy(CANDIDATES)
                pins["pickr"]["repo"] = repo
                with self.subTest(repo=repo), self.assertRaisesRegex(
                        ValueError, "previously observed"):
                    screen_pins_against_history(pins, ["synthetic-no-file-read"])

    def test_receipt_explicitly_denies_any_training_or_ood_qualification(self):
        receipt = public_source_floor_receipt()
        self.assertEqual(receipt["public_distinct_original_source_families_blocked"], 27)
        self.assertFalse(receipt["complete_historical_source_inventory_proven"])
        self.assertFalse(receipt["complete_previous_png_rgba_registry_proven"])
        self.assertFalse(receipt["training_approved"])
        self.assertFalse(receipt["ood_evaluation_approved"])
        self.assertEqual(receipt["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()

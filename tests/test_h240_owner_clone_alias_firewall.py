"""H240 source-only owner Git clone alias firewall tests (synthetic Git fixtures).

No private source names, media, gold, OOD, labels, checkpoints or network.
"""
from pathlib import Path
import tempfile
import unittest

from h178.owner_git_inventory import (
    _github_root, audit_source_inventory, inventory_local_original_sources,
)
from test_h178_owner_git_inventory import _synthetic_original, fixture


class H240OwnerGitFamilyCanonicalization(unittest.TestCase):
    def test_syntactically_different_github_origins_are_one_family(self):
        for origin in (
            "https://github.com/mock-owner/used-board",
            "https://github.com/mock-owner/used-board.git",
            "https://github.com/mock-owner/used-board.git.git",
            "https://github.com/MOCK-OWNER/USED-BOARD.GIT.GIT/",
            "git@github.com:mock-owner/used-board.git",
            "git@github.com:MOCK-OWNER/USED-BOARD.GIT.GIT",
        ):
            with self.subTest(origin_type="ssh" if origin.startswith("git@") else "https"):
                self.assertEqual(_github_root(origin), "mock-owner/used-board")

    def test_invalid_and_unsafe_origins_rejected(self):
        for origin in (
            "http://github.com/mock-owner/used-board.git",
            "https://token@github.com/mock-owner/used-board.git",
            "https://github.com.evil.example/mock-owner/used-board",
            "https://github.com/mock-owner/.git",
            "https://github.com/mock-owner/..git",
            "https://github.com/mock-owner/used-board.git/other",
            "file:///tmp/mock-owner/used-board.git",
        ):
            with self.subTest(origin_type=origin.split(":")[0]), self.assertRaises(ValueError):
                _github_root(origin)

    def test_owner_inventory_coalesces_all_clone_aliases_same_git_revision(self):
        for remote in (
            "https://github.com/mock-owner/used-board.git.git",
            "git@github.com:MOCK-OWNER/USED-BOARD.GIT.GIT",
        ):
            with self.subTest(remote_scheme=remote.split(":")[0]):
                with tempfile.TemporaryDirectory(prefix="h240-source-only-") as tmp:
                    phase = Path(tmp) / "private-h140"
                    _source, git_sha = _synthetic_original(phase, origin=remote)
                    receipt = inventory_local_original_sources([tmp])
                    self.assertEqual(receipt["source_checkouts_inspected"], 1)
                    self.assertEqual(receipt["distinct_repositories"], 1)
                    self.assertEqual(
                        receipt["_owner_only_repository_shas"],
                        frozenset({("mock-owner/used-board", git_sha)}),
                    )
                    self.assertFalse(receipt["historical_source_completeness_proven"])
                    self.assertFalse(receipt["model_training_approved"])
                    self.assertFalse(receipt["ood_evaluation_approved"])

    def test_owner_historical_checkout_reuse_rejected_before_any_admission(self):
        for remote in (
            "https://github.com/mock-owner/used-board.git.git",
            "git@github.com:MOCK-OWNER/USED-BOARD.GIT.GIT",
        ):
            for incoming in (
                "mock-owner/used-board",
                "MOCK-OWNER/USED-BOARD.GIT",
                "mock-owner/used-board.git.git",
            ):
                with self.subTest(version=remote.split(":")[0], candidate=incoming):
                    with tempfile.TemporaryDirectory(prefix="h240-owner-git-") as tmp:
                        _synthetic_original(Path(tmp) / "private-h140", origin=remote)
                        meta, pixels, history, paired = fixture()
                        meta["sources"][0]["repo"] = incoming
                        with self.assertRaisesRegex(ValueError, "already seen privately"):
                            audit_source_inventory(meta, pixels, history, paired, roots=[tmp])

    def test_candidate_even_with_prior_registry_alias_is_never_admitted(self):
        from test_h178_owner_git_inventory import _attest_dummy_history
        with tempfile.TemporaryDirectory(prefix="h240-owner-history-") as tmp:
            _source, sha = _synthetic_original(
                Path(tmp) / "private-h140",
                origin="https://github.com/mock-owner/used-board.git.git",
            )
            args = _attest_dummy_history(fixture(), "mock-owner/used-board.git", sha)
            receipt = audit_source_inventory(*args, roots=[tmp])
            self.assertFalse(receipt["source_history_completeness_proven"])
            self.assertFalse(receipt["training_approved"])
            self.assertFalse(receipt["ood_evaluation_approved"])
            self.assertEqual(receipt["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()

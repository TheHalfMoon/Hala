"""H178 tests use disposable local Git repositories, no network or old media."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from h178.owner_git_inventory import (
    inventory_local_original_sources, audit_source_inventory,
)
from test_h177_public_origin_pins import fixture
from h178.scan_owner_sources import run as local_cli


def _run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if proc.returncode:
        raise RuntimeError("synthetic Git test setup failed")
    return proc.stdout.strip()


def _synthetic_original(phase_root: Path, *,
                        origin: str = "https://github.com/mock-owner/used-board.git"):
    source = phase_root / "upstream" / "used-board"
    source.mkdir(parents=True)
    _run_git(source, "init", "-q")
    _run_git(source, "config", "core.autocrlf", "false")
    _run_git(source, "remote", "add", "origin", origin)
    (source / "README.txt").write_text(
        "synthetic Git source identity only, not a benchmark", encoding="utf-8"
    )
    _run_git(source, "add", "README.txt")
    _run_git(source, "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid",
             "commit", "-q", "-m", "dummy synthetic original source")
    return source, _run_git(source, "rev-parse", "HEAD")


def _attest_dummy_history(inputs: tuple, repo: str, sha: str):
    meta, pixels, history, attribution = deepcopy(inputs)
    row = {
        "app": "prior-synthetic", "repo": repo, "git_sha": sha,
        "html_sha256": "f"*64,
    }
    attribution["families"].append(row)
    history["historical_app_ids"].append(row["app"])
    history["historical_repositories"].append(row["repo"])
    history["source_git_shas"].append(row["git_sha"])
    history["source_html_shas"].append(row["html_sha256"])
    return meta, pixels, history, attribution


class OwnerGitOriginInventoryTests(unittest.TestCase):
    def test_only_approved_historical_git_source_folders_scanned(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            source, sha = _synthetic_original(phase)
            # Media/gold are NOT accessed: deliberately place non-source
            # data outside shallow upstream/NAME/.git traversal.
            (phase / "ood-32-v3").mkdir()
            (phase / "ood-32-v3" / "gold.json").write_text(
                "NEVER_OPEN", encoding="utf-8"
            )
            result = inventory_local_original_sources([Path(root)])
            self.assertEqual(result["source_checkouts_inspected"], 1)
            self.assertEqual(result["distinct_repositories"], 1)
            self.assertIn(
                ("mock-owner/used-board", sha),
                result["_owner_only_repository_shas"],
            )
            self.assertFalse(result["historical_source_completeness_proven"])
            self.assertFalse(result["model_training_approved"])
            self.assertEqual(len(result["source_identity_fingerprint_sha256"]), 64)

    def test_phased_root_and_parent_root_produce_same_census(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h140"
            _synthetic_original(phase)
            a = inventory_local_original_sources([Path(root)])
            b = inventory_local_original_sources([phase])
            self.assertEqual(
                a["source_identity_fingerprint_sha256"],
                b["source_identity_fingerprint_sha256"],
            )

    def test_invalid_remote_host_and_scheme_fail_closed(self):
        for origin in ("https://evil.example/org/repo.git",
                       "http://github.com/org/repo.git",
                       "https://token@github.com/org/repo.git",
                       "file:///outside/private/repo",
                       "https://github.com/org/repo/third"):
            with self.subTest(origin_class=origin.split(":")[0]):
                with tempfile.TemporaryDirectory() as root:
                    _synthetic_original(Path(root) / "private-h139",
                                        origin=origin)
                    with self.assertRaises(ValueError):
                        inventory_local_original_sources([Path(root)])

    def test_no_old_git_source_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "private-h137" / "upstream").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no verified original Git"):
                inventory_local_original_sources([root])

    def test_duplicate_phase_root_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h140"
            _synthetic_original(phase)
            with self.assertRaisesRegex(ValueError, "duplicate historical"):
                inventory_local_original_sources([phase, phase])

    def test_local_discovered_repo_missing_from_owner_history_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            _synthetic_original(phase)
            with self.assertRaisesRegex(ValueError, "omits a locally observed"):
                audit_source_inventory(*fixture(), roots=[root])

    def test_local_observed_repo_masquerading_as_fresh_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            _synthetic_original(phase)
            meta, pixels, old, paired = fixture()
            meta["sources"][0]["repo"] = "MOCK-OWNER/USED-BOARD"
            with self.assertRaisesRegex(ValueError, "already seen privately"):
                audit_source_inventory(
                    meta, pixels, old, paired, roots=[root]
                )

    def test_paired_owner_history_includes_local_source_but_no_admission(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h140"
            _source, sha = _synthetic_original(phase)
            args = _attest_dummy_history(
                fixture(), "mock-owner/used-board", sha
            )
            result = audit_source_inventory(*args, roots=[root])
            self.assertEqual(result["source_checkouts_inspected"], 1)
            self.assertFalse(result["source_history_completeness_proven"])
            self.assertFalse(result["training_approved"])
            self.assertFalse(result["ood_evaluation_approved"])
            self.assertEqual(result["model_calls"], 0)

    def test_inventory_cli_omits_owner_paths_and_remote_names(self):
        with tempfile.TemporaryDirectory() as root:
            _synthetic_original(Path(root) / "private-h139")
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = local_cli(["--historical-root", root])
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertIn("source_checkouts=1", out.getvalue())
            self.assertIn("HISTORY_COMPLETENESS_UNPROVEN", out.getvalue())
            self.assertNotIn(root, out.getvalue())
            self.assertNotIn("mock-owner", out.getvalue())

    def test_full_owner_cli_checks_only_source_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            _src, sha = _synthetic_original(Path(root) / "private-h140")
            data = _attest_dummy_history(
                fixture(), "mock-owner/used-board", sha
            )
            paths = [Path(root) / ("source-%s.json" % n) for n in range(4)]
            for path, obj in zip(paths, data):
                path.write_text(json.dumps(obj), encoding="utf-8")
            flags = ["--historical-root", root]
            for label, path in zip((
                "--candidates", "--prior-pixel-hashes",
                "--prior-source-history", "--paired-source-history",
            ), paths):
                flags.extend([label, str(path)])
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = local_cli(flags)
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertIn("H178_OWNER_GIT_ORIGINS_ONLY", out.getvalue())
            self.assertIn("HISTORY_COMPLETENESS_UNPROVEN", out.getvalue())
            self.assertNotIn(root, out.getvalue())

    def test_owner_cli_partial_metadata_fails_privately(self):
        with tempfile.TemporaryDirectory() as root:
            _synthetic_original(Path(root) / "private-h140")
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = local_cli([
                    "--historical-root", root,
                    "--candidates", str(Path(root) / "fake-private.json"),
                ])
            self.assertEqual(code, 2)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("H178_FAIL_CLOSED", err.getvalue())
            self.assertNotIn(root, err.getvalue())

    def test_broken_git_pointer_and_source_store_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            fake_source = phase / "upstream" / "synthetic-missing-git"
            fake_source.mkdir(parents=True)
            try:
                (fake_source / ".git").symlink_to(
                    Path(root) / "missing-git-object", target_is_directory=True
                )
            except (OSError, NotImplementedError):
                return  # Windows privilege restriction; Linux CI exercises it.
            with self.assertRaisesRegex(ValueError, "original \\.git cannot"):
                inventory_local_original_sources([root])
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            phase.mkdir()
            (phase / "upstream").symlink_to(
                Path(root) / "absent-upstream-store", target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "unsafe historical source store"):
                inventory_local_original_sources([root])

    def test_invalid_root_and_symlink_does_not_escape_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            phase = Path(root) / "private-h139"
            _synthetic_original(phase)
            with self.assertRaises(ValueError):
                inventory_local_original_sources(["not/an/absolute/path"])
            link = Path(root) / "private-h140"
            try:
                link.symlink_to(phase, target_is_directory=True)
            except (OSError, NotImplementedError):
                return  # Windows policy may prohibit test symlink creation.
            with self.assertRaises(ValueError):
                inventory_local_original_sources([link])


if __name__ == "__main__":
    unittest.main()

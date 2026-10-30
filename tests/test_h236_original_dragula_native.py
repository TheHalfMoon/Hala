"""H236 source-only synthetic provenance QA, no owner media or network."""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import h236.original_dragula_native as qa


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True,
        text=True, check=True, timeout=15,
    )
    return result.stdout.strip()


class H236OriginalDragulaSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hala-h236-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "synthetic-original"
        self.root.mkdir()
        inputs = {
            "license": b"The MIT License, SYNTHETIC TEST ONLY\n",
            "dist/dragula.min.js": b"/* synthetic JS stub, no donor media */\n",
            "index.html": b"<!doctype html><title>Synthetic test source</title>\n",
        }
        for relative, content in inputs.items():
            target = self.root.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        self.hashes = {name: sha256(content).hexdigest()
                       for name, content in inputs.items()}
        git(self.root, "init", "-q")
        git(self.root, "remote", "add", "origin",
            "https://github.com/bevacqua/dragula.git")
        git(self.root, "add", ".")
        git(self.root, "-c", "user.name=H236 Synthetic",
            "-c", "user.email=h236@example.invalid", "commit", "-q",
            "-m", "Synthetic source preflight")
        self.commit = git(self.root, "rev-parse", "HEAD")
        self.out = Path(self.tmp.name) / "private-evidence"
        self.p1 = patch.object(qa, "COMMIT", self.commit)
        self.p2 = patch.object(qa, "PINNED", self.hashes)
        self.p1.start()
        self.p2.start()
        self.addCleanup(self.p1.stop)
        self.addCleanup(self.p2.stop)

    def test_pinned_clean_synthetic_source_preflight(self):
        checked = qa.preflight(self.root, self.out)
        self.assertEqual(checked["original_source"], self.root.resolve())
        self.assertFalse(self.out.exists())

    def test_original_git_sha_mismatch_denied(self):
        with patch.object(qa, "COMMIT", "f" * 40):
            with self.assertRaisesRegex(ValueError, "Git SHA mismatch"):
                qa.preflight(self.root, self.out)

    def test_original_untracked_file_denied(self):
        (self.root / "extra.png").write_bytes(b"not-a-real-png")
        with self.assertRaisesRegex(ValueError, "clean"):
            qa.preflight(self.root, self.out)

    def test_wrong_sha_denied_even_if_git_checkout_clean(self):
        changed = dict(self.hashes)
        changed["dist/dragula.min.js"] = "0" * 64
        with patch.object(qa, "PINNED", changed):
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                qa.preflight(self.root, self.out)

    def test_non_github_origin_denied(self):
        git(self.root, "remote", "set-url", "origin",
            "https://example.invalid/bevacqua/dragula.git")
        with self.assertRaisesRegex(ValueError, "remote identity"):
            qa.preflight(self.root, self.out)

    def test_unrelated_github_origin_denied(self):
        git(self.root, "remote", "set-url", "origin",
            "https://github.com/unknown/other.git")
        with self.assertRaisesRegex(ValueError, "remote identity"):
            qa.preflight(self.root, self.out)

    def test_ssh_equivalent_original_origin_accepted(self):
        git(self.root, "remote", "set-url", "origin",
            "git@github.com:bevacqua/dragula.git")
        self.assertEqual(qa.preflight(self.root, self.out)["output"],
                         self.out.resolve())

    def test_known_historical_git_family_denied(self):
        for original in ("choices-js/choices", "flatpickr/flatpickr",
                         "leongersen/nouislider", "leaverou/awesomplete",
                         "nudeui/awesomplete", "sortablejs/sortable",
                         "robinparisi/tingle"):
            with self.subTest(original=original), patch.object(qa, "REPO", original):
                with self.assertRaisesRegex(ValueError, "historical"):
                    qa.preflight(self.root, self.out)

    def test_output_inside_donor_denied(self):
        with self.assertRaisesRegex(ValueError, "leave original"):
            qa.preflight(self.root, self.root / "private")

    def test_output_inside_other_git_checkout_denied(self):
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        git(other, "init", "-q")
        with self.assertRaisesRegex(ValueError, "outside all Git"):
            qa.preflight(self.root, other / "evidence")

    def test_output_must_be_create_only(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, "create-only"):
            qa.preflight(self.root, self.out)

    def test_missing_mit_license_denied(self):
        (self.root / "license").write_text("not licensed\n")
        git(self.root, "add", "license")
        git(self.root, "-c", "user.name=H236 Synthetic",
            "-c", "user.email=h236@example.invalid", "commit", "-q",
            "-m", "Synthetic non MIT")
        with patch.object(qa, "COMMIT", git(self.root, "rev-parse", "HEAD")):
            new_digests = dict(self.hashes)
            new_digests["license"] = sha256((self.root / "license").read_bytes()).hexdigest()
            with patch.object(qa, "PINNED", new_digests):
                with self.assertRaisesRegex(ValueError, "MIT license"):
                    qa.preflight(self.root, self.out)

    def test_research_contract_is_not_a_training_release(self):
        self.assertEqual(qa.REPO, "bevacqua/dragula")
        # setUp intentionally substitutes a synthetic local Git commit and
        # synthetic file SHA pins, so the actual upstream pin is verified
        # separately by the owner native replay, not here.
        self.assertEqual(qa.COMMIT, self.commit)
        self.assertEqual(set(qa.PINNED), {"license", "dist/dragula.min.js", "index.html"})
        self.assertNotIn("H173", qa.FIXTURE)
        self.assertNotIn("https:", qa.FIXTURE)


if __name__ == "__main__":
    unittest.main()

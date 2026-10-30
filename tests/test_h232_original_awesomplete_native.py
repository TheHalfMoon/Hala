"""H232 source-only adversarial tests with synthetic files, no original media."""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import h232.original_awesomplete_native as qa

def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args],
                       capture_output=True, text=True, check=True, timeout=15)
    return p.stdout.strip()

class H232PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hala-h232-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "donor"
        self.root.mkdir()
        data = {
            "LICENSE": b"MIT License\nsynthetic only\n",
            "test/fixtures/options.html": b"<input id='synthetic'>",
            "awesomplete.js": b"/* synthetic JS, never executed */",
        }
        for name, value in data.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(value)
        self.hashes = {k: sha256(v).hexdigest() for k, v in data.items()}
        git(self.root, "init", "-q")
        git(self.root, "remote", "add", "origin", "https://github.com/LeaVerou/awesomplete.git")
        git(self.root, "add", ".")
        git(self.root, "-c", "user.name=H232 Synthetic", "-c",
            "user.email=synthetic@example.invalid", "commit", "-q", "-m", "synthetic")
        self.sha = git(self.root, "rev-parse", "HEAD")
        self.output = Path(self.tmp.name) / "owner-private-output"
        self.p1 = patch.object(qa, "COMMIT", self.sha)
        self.p2 = patch.object(qa, "PINNED", self.hashes)
        self.p1.start()
        self.p2.start()
        self.addCleanup(self.p1.stop)
        self.addCleanup(self.p2.stop)

    def test_synthetic_clean_preflight(self):
        result = qa.preflight(self.root, self.output)
        self.assertEqual(result["root"], self.root.resolve())
        self.assertFalse(self.output.exists())

    def test_dirty_original_denied(self):
        (self.root / "awesomplete.js").write_text("changed")
        with self.assertRaisesRegex(ValueError, "dirty"):
            qa.preflight(self.root, self.output)

    def test_wrong_original_sha_denied(self):
        with patch.object(qa, "COMMIT", "0" * 40):
            with self.assertRaisesRegex(ValueError, "Git SHA"):
                qa.preflight(self.root, self.output)

    def test_wrong_upstream_denied(self):
        git(self.root, "remote", "set-url", "origin", "https://github.com/attacker/fake.git")
        with self.assertRaisesRegex(ValueError, "upstream"):
            qa.preflight(self.root, self.output)

    def test_modified_original_asset_denied(self):
        (self.root / "awesomplete.js").write_text("changed")
        git(self.root, "add", ".")
        git(self.root, "-c", "user.name=H232 Synthetic", "-c",
            "user.email=synthetic@example.invalid", "commit", "-q", "-m", "changed")
        with patch.object(qa, "COMMIT", git(self.root, "rev-parse", "HEAD")):
            with self.assertRaisesRegex(ValueError, "byte mismatch"):
                qa.preflight(self.root, self.output)

    def test_existing_output_denied(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "create-only"):
            qa.preflight(self.root, self.output)

    def test_evidence_inside_source_denied(self):
        with self.assertRaisesRegex(ValueError, "outside original source"):
            qa.preflight(self.root, self.root / "private")

    def test_evidence_inside_other_git_checkout_denied(self):
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        git(other, "init", "-q")
        with self.assertRaisesRegex(ValueError, "outside all Git"):
            qa.preflight(self.root, other / "evidence")

    def test_known_historical_family_denied(self):
        with patch.object(qa, "REPO", "tastejs/todomvc"):
            with self.assertRaisesRegex(ValueError, "historical"):
                qa.preflight(self.root, self.output)

if __name__ == "__main__":
    unittest.main()

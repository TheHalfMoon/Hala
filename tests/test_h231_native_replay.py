"""H231 synthetic-only regression; never owner OOD, gold, or optimizer data."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from h231.native_replay import SCHEMA, _allowed_url, preflight, replay


HTML = """<!doctype html><meta charset='utf-8'><title>New synthetic QA only</title>
<style>body{font:20px Arial;padding:40px} #card,#drop{display:inline-block;
padding:22px;border:1px solid black;margin:12px}</style>
<div id='state' data-clicked='no' data-typed='no' data-dropped='no'></div>
<button id='go' onclick="document.querySelector('#state').dataset.clicked='yes'">Confirm</button>
<input id='entry' aria-label='Text' oninput="document.querySelector('#state').dataset.typed=
(this.value==='مرحبا'?'yes':'no')">
<div id='card' draggable='true'>Drag card</div>
<div id='drop' ondragover='event.preventDefault()' ondrop="event.preventDefault();
document.querySelector('#state').dataset.dropped='yes'">Drop here</div>
"""
CASES = [
    {"id": "click-en", "lang": "en", "action": "click", "selector": "#go",
     "destination": "", "text": "", "expect_selector": "#state",
     "expect_attribute": "data-clicked", "expect_value": "yes"},
    {"id": "type-ar", "lang": "ar", "action": "type", "selector": "#entry",
     "destination": "", "text": "مرحبا", "expect_selector": "#state",
     "expect_attribute": "data-typed", "expect_value": "yes"},
    {"id": "drag-en", "lang": "en", "action": "drag", "selector": "#card",
     "destination": "#drop", "text": "", "expect_selector": "#state",
     "expect_attribute": "data-dropped", "expect_value": "yes"},
]


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       text=True, timeout=15, check=True)
    return p.stdout.strip()


class H231SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hala-h231-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "independent-new-source"
        self.source.mkdir()
        (self.source / "fixture.html").write_text(HTML, encoding="utf-8")
        (self.source / "LICENSE").write_text("ORIGINAL TEST-ONLY FIXTURE, NOT THIRD-PARTY LICENSE\n", encoding="utf-8")
        git(self.source, "init", "-q")
        git(self.source, "remote", "add", "origin",
            "https://github.com/hala-test/new-synthetic-fixture.git")
        git(self.source, "add", "fixture.html", "LICENSE")
        git(self.source, "-c", "user.name=H231 QA", "-c", "user.email=synthetic@example.invalid",
            "commit", "-q", "-m", "original synthetic GUI fixture")
        self.manifest = {
            "schema": SCHEMA, "repo": "hala-test/new-synthetic-fixture",
            "git_sha": git(self.source, "rev-parse", "HEAD"),
            "license": "LICENSE",
            "license_sha256": sha256((self.source / "LICENSE").read_bytes()).hexdigest(),
            "html": "fixture.html",
            "html_sha256": sha256((self.source / "fixture.html").read_bytes()).hexdigest(),
            "cases": copy.deepcopy(CASES),
        }
        self.output = self.base / "private-evidence"

    def test_preflight_without_browser(self):
        result = preflight(self.manifest, self.source, self.output)
        self.assertEqual(result["commit"], self.manifest["git_sha"])
        self.assertFalse(self.output.exists())

    def test_dirty_source_denied(self):
        with (self.source / "fixture.html").open("a", encoding="utf-8") as file:
            file.write("changed")
        with self.assertRaisesRegex(ValueError, "dirty"):
            preflight(self.manifest, self.source, self.output)

    def test_bad_sha_denied(self):
        self.manifest["html_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "html_sha256"):
            preflight(self.manifest, self.source, self.output)

    def test_forged_origin_claim_fails_even_with_identical_git_sha(self):
        self.manifest["repo"] = "invented-owner/renamed-source"
        with self.assertRaisesRegex(ValueError, "match origin"):
            preflight(self.manifest, self.source, self.output)

    def test_offsite_git_origin_denied(self):
        git(self.source, "remote", "set-url", "origin",
            "https://example.invalid/hala-test/new-synthetic-fixture.git")
        with self.assertRaisesRegex(ValueError, "Git origin must be"):
            preflight(self.manifest, self.source, self.output)

    def test_git_clone_alias_of_same_origin_accepted(self):
        git(self.source, "remote", "set-url", "origin",
            "git@github.com:HALA-TEST/NEW-SYNTHETIC-FIXTURE.GIT")
        self.assertEqual(
            preflight(self.manifest, self.source, self.output)["repo"],
            "hala-test/new-synthetic-fixture",
        )

    def test_no_evidence_in_source_repo(self):
        with self.assertRaisesRegex(ValueError, "outside source"):
            preflight(self.manifest, self.source, self.source / "private")

    def test_reject_historical_original_family(self):
        self.manifest["repo"] = "tastejs/todomvc"
        with self.assertRaisesRegex(ValueError, "historical original"):
            preflight(self.manifest, self.source, self.output)

    def test_duplicate_cases_denied(self):
        self.manifest["cases"].append(copy.deepcopy(CASES[0]))
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            preflight(self.manifest, self.source, self.output)

    def test_no_traversal(self):
        self.manifest["html"] = "../private.html"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            preflight(self.manifest, self.source, self.output)

    def test_only_scoped_file_uris(self):
        self.assertTrue(_allowed_url((self.source / "fixture.html").as_uri(), self.source.resolve()))
        self.assertFalse(_allowed_url("https://example.com/private", self.source.resolve()))
        self.assertFalse(_allowed_url((self.base / "elsewhere.png").as_uri(), self.source.resolve()))

    def test_browser_three_actual_native_operations(self):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                if not Path(p.chromium.executable_path).exists() and not edge.is_file():
                    self.skipTest("Hosted machine has no Chromium or Edge")
        except ImportError:
            self.skipTest("Optional Playwright not installed on hosted runner")
        manifest_file = self.base / "source-manifest.json"
        manifest_file.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
        result = replay(manifest_file, self.source, self.output)
        self.assertTrue(result["all_native_oracle_outcomes_passed"])
        self.assertEqual(result["network_or_unscoped_file_attempts"], 0)
        self.assertEqual(len(result["cases"]), 3)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["training_approved"])
        self.assertFalse(result["complete_historical_source_and_rgba_proven"])
        for row in result["cases"]:
            self.assertTrue(row["native_outcome_matches_oracle"])
            self.assertEqual(len(row["before_rgba_sha256"]), 64)
            self.assertTrue((self.output / (row["id"] + "-before.png")).exists())
            self.assertTrue((self.output / (row["id"] + "-after.png")).exists())
        self.assertTrue((self.output / "report.json").is_file())
        self.assertFalse((self.output / "INCOMPLETE_NO_ADMISSION").exists())


    def test_output_under_another_git_repo_denied(self):
        other = self.base / "other-checkout"
        other.mkdir()
        git(other, "init", "-q")
        with self.assertRaisesRegex(ValueError, "outside every Git checkout"):
            preflight(self.manifest, self.source, other / "evidence")

    def test_remote_subresource_is_denied_and_evidence_marked_incomplete(self):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                if not Path(p.chromium.executable_path).exists() and not edge.is_file():
                    self.skipTest("Hosted browser unavailable")
        except ImportError:
            self.skipTest("Optional Playwright unavailable")
        with (self.source / "fixture.html").open("a", encoding="utf-8") as file:
            file.write('<img src="https://example.invalid/forbidden.png">')
        git(self.source, "add", "fixture.html")
        git(self.source, "-c", "user.name=H231 QA", "-c", "user.email=synthetic@example.invalid",
            "commit", "-q", "-m", "synthetic hostile external subresource")
        self.manifest["git_sha"] = git(self.source, "rev-parse", "HEAD")
        self.manifest["html_sha256"] = sha256((self.source / "fixture.html").read_bytes()).hexdigest()
        self.manifest["cases"] = [copy.deepcopy(CASES[0])]
        manifest_file = self.base / "source-manifest.json"
        manifest_file.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "network or out-of-root"):
            replay(manifest_file, self.source, self.output)
        self.assertTrue((self.output / "INCOMPLETE_NO_ADMISSION").exists())
        self.assertFalse((self.output / "report.json").exists())

    def test_wrong_native_oracle_outcome_fails_closed(self):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                if not Path(p.chromium.executable_path).exists() and not edge.is_file():
                    self.skipTest("Hosted browser unavailable")
        except ImportError:
            self.skipTest("Optional Playwright unavailable")
        self.manifest["cases"] = [copy.deepcopy(CASES[0])]
        self.manifest["cases"][0]["expect_value"] = "never-matches"
        manifest_file = self.base / "source-manifest.json"
        manifest_file.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "native oracle outcome mismatch"):
            replay(manifest_file, self.source, self.output)
        self.assertTrue((self.output / "INCOMPLETE_NO_ADMISSION").exists())
        self.assertFalse((self.output / "report.json").exists())

if __name__ == "__main__":
    unittest.main()

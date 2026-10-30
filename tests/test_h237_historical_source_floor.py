"""H237: source-family leak regression from H168/H169/H171/H172 public evidence.

Public source metadata and synthetic Git fixture ONLY. No owner images/gold,
no existing OOD access and no data/model admission.
"""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
import unittest

from h174.split_firewall import (
    FROZEN_PRIOR_ORIGINAL_REPOS,
    HISTORY_SCHEMA,
    SCHEMA as SPLIT_SCHEMA,
    canonical_repository_family,
    validate,
)
from h231.native_replay import SCHEMA as CAPTURE_SCHEMA, preflight


# Historical ORIGINAL donors, even under different commits or app names.
# Two Awesomplete slugs identify the SAME immutable GitHub repository ID 30987756.
OLD_GIT_FAMILIES = (
    "Choices-js/Choices", "choices-js/choices.git",
    "leongersen/noUiSlider", "LEONGERSEN/NOuiSlider.git",
    "flatpickr/flatpickr", "FLATPICKR/FLATPICKR.git",
    "LeaVerou/awesomplete", "nudeui/awesomplete",
    "nudeui/awesomplete.git",
)


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True,
        text=True, check=True, timeout=15)
    return process.stdout.strip()


def metadata_for(repo: str) -> tuple[dict, dict]:
    metadata = {
        "schema": SPLIT_SCHEMA,
        "sources": [
            {"app": "apparentlynovel", "split": "TRAIN", "repo": repo,
             "git_sha": "1"*40, "html_sha256": "1"*64, "license": "MIT"},
            {"app": "secondfresh", "split": "DEV", "repo": "synthetic/new-dev",
             "git_sha": "2"*40, "html_sha256": "2"*64, "license": "MIT"},
            {"app": "oodhold", "split": "OOD_RESERVED", "repo": "synthetic/new-hold",
             "git_sha": "3"*40, "html_sha256": "3"*64, "license": "MIT"},
        ],
        "cases": [
            {"index": 0, "app": "apparentlynovel", "split": "TRAIN",
             "image_sha256": "4"*64, "visual_rgba_sha256": "5"*64,
             "language": "en", "action_type": "type",
             "instruction": "Enter clearly visible text"},
            {"index": 1, "app": "secondfresh", "split": "DEV",
             "image_sha256": "6"*64, "visual_rgba_sha256": "7"*64,
             "language": "ar", "action_type": "drag",
             "instruction": "اسحب العنصر إلى الهدف"},
        ],
    }
    history = {"schema": HISTORY_SCHEMA, "image_shas": ["8"*64],
               "visual_shas": ["9"*64]}
    return metadata, history


class H237OldOriginalSourceFloor(unittest.TestCase):
    def test_exact_public_source_history_aliases_are_frozen(self):
        self.assertEqual(
            FROZEN_PRIOR_ORIGINAL_REPOS,
            frozenset({
                "choices-js/choices", "leongersen/nouislider",
                "flatpickr/flatpickr", "leaverou/awesomplete",
                "nudeui/awesomplete",
            }),
        )
        for name in OLD_GIT_FAMILIES:
            with self.subTest(name=name):
                self.assertIn(canonical_repository_family(name),
                              FROZEN_PRIOR_ORIGINAL_REPOS)

    def test_old_original_family_denied_under_fresh_app_id_and_forged_sha(self):
        for name in OLD_GIT_FAMILIES:
            with self.subTest(name=name):
                meta, old = metadata_for(name)
                with self.assertRaisesRegex(ValueError, "previous original TRAIN/DEV"):
                    validate(meta, old)

    def test_history_cannot_hide_in_new_dev_or_ood_reservation(self):
        for split in ("DEV", "OOD_RESERVED"):
            for name in ("Choices-js/Choices", "flatpickr/flatpickr",
                         "nudeui/awesomplete"):
                meta, registry = metadata_for("synthetic/fresh-train")
                source = next(s for s in meta["sources"] if s["split"] == split)
                source["repo"] = name
                with self.subTest(name=name, split=split), self.assertRaisesRegex(
                        ValueError, "previous original TRAIN/DEV"):
                    validate(meta, registry)

    def test_unrelated_synthetic_metadata_still_prep_parses(self):
        meta, registry = metadata_for("synthetic/fresh-train")
        result = validate(meta, registry)
        self.assertEqual(result["train_cases"], 1)
        self.assertEqual(result["dev_cases"], 1)
        self.assertEqual(result["model_calls"], 0)
        self.assertNotIn("training_approved", result)

    def test_native_capture_preflight_denies_old_repos_even_if_git_origin_matches(self):
        with tempfile.TemporaryDirectory(prefix="h237-source-only-") as tmp:
            root = Path(tmp) / "new-original"
            root.mkdir()
            (root / "index.html").write_text(
                "<!doctype html><title>synthetic only</title>", encoding="utf-8")
            (root / "LICENSE").write_text(
                "synthetic fake fixture, no original media", encoding="utf-8")
            _git(root, "init", "-q")
            _git(root, "add", "index.html", "LICENSE")
            _git(root, "-c", "user.name=H237 Synthetic",
                 "-c", "user.email=h237@example.invalid",
                 "commit", "-q", "-m", "new synthetic Git fixture")
            head = _git(root, "rev-parse", "HEAD")
            m = {
                "schema": CAPTURE_SCHEMA,
                "repo": "synthetic/new-source", "git_sha": head,
                "license": "LICENSE",
                "license_sha256": sha256((root / "LICENSE").read_bytes()).hexdigest(),
                "html": "index.html",
                "html_sha256": sha256((root / "index.html").read_bytes()).hexdigest(),
                "cases": [{
                    "id": "click-test", "lang": "en", "action": "click",
                    "selector": "#go", "destination": "", "text": "",
                    "expect_selector": "#state", "expect_attribute": "data-ready",
                    "expect_value": "yes",
                }],
            }
            for name in OLD_GIT_FAMILIES:
                m["repo"] = name
                _git(root, "remote", "remove", "origin") if "origin" in _git(
                    root, "remote").splitlines() else None
                _git(root, "remote", "add", "origin",
                     "https://github.com/" + name.removesuffix(".git") + ".git")
                with self.subTest(name=name), self.assertRaisesRegex(
                        ValueError, "known historical original"):
                    preflight(m, root, Path(tmp) / "private-output")
            m["repo"] = "synthetic/new-source"
            _git(root, "remote", "set-url", "origin",
                 "https://github.com/synthetic/new-source.git")
            self.assertEqual(
                preflight(m, root, Path(tmp) / "private-output")["repo"],
                "synthetic/new-source",
            )


if __name__ == "__main__":
    unittest.main()

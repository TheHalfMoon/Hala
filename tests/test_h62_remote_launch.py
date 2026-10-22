"""H62 the pinned remote entry point: the fail-closed properties that matter.

A training run that waited hours for a free GPU must fail closed rather than
train something other than the reviewed source, and a failed child must
never be reported as a successful orchestration. These tests exercise both
properties without contacting any provider.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h62 import remote_h62_launch as launch


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RuntimeAdmissionTests(unittest.TestCase):
    def _environment(self, **overrides):
        environment = {
            "COLAB_RELEASE_TAG": "test-release",
            "HALA_FREE_COLAB_RESEARCH": "1",
            "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
        }
        environment.update(overrides)
        return environment

    def test_a_missing_workspace_is_refused(self):
        with mock.patch.dict(os.environ, self._environment(), clear=True):
            with mock.patch.object(Path, "exists", return_value=False):
                with self.assertRaises(RuntimeError):
                    launch.require_free_colab_runtime()

    def test_a_session_that_does_not_identify_as_colab_is_refused(self):
        with mock.patch.dict(os.environ, self._environment(COLAB_RELEASE_TAG=""), clear=True):
            with mock.patch.object(Path, "exists", return_value=True):
                with self.assertRaises(RuntimeError):
                    launch.require_free_colab_runtime()

    def test_the_free_tier_confirmation_is_required(self):
        for missing in ("HALA_FREE_COLAB_RESEARCH", "HALA_NO_PAID_RUNTIME_CONFIRMED"):
            with mock.patch.dict(os.environ, self._environment(**{missing: ""}), clear=True):
                with mock.patch.object(Path, "exists", return_value=True):
                    with self.assertRaises(RuntimeError):
                        launch.require_free_colab_runtime()

    def test_a_kaggle_notebook_is_refused(self):
        with mock.patch.dict(os.environ, self._environment(KAGGLE_KERNEL_RUN_TYPE="Batch"),
                             clear=True):
            with mock.patch.object(Path, "exists", return_value=True):
                with self.assertRaises(RuntimeError):
                    launch.require_free_colab_runtime()

    def test_a_confirmed_free_colab_session_is_admitted(self):
        with mock.patch.dict(os.environ, self._environment(), clear=True):
            with mock.patch.object(Path, "exists", return_value=True):
                admitted = launch.require_free_colab_runtime()
                self.assertTrue(admitted["free_runtime_confirmed"])
                self.assertEqual(admitted["colab_release_tag"], "test-release")


class PinnedSourceTests(unittest.TestCase):
    def test_an_invalid_sha_is_refused_before_any_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(launch.subprocess, "run") as fake:
                with self.assertRaises(RuntimeError):
                    launch.fetch_pinned_source(Path(directory), "not-a-sha")
                self.assertEqual(fake.call_count, 0)

    def test_a_head_that_is_not_the_pinned_sha_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = "a" * 40
            responses = [
                FakeCompleted(0, ""),                       # git init
                FakeCompleted(0, ""),                       # remote add
                FakeCompleted(0, ""),                       # fetch
                FakeCompleted(0, ""),                       # checkout
                FakeCompleted(0, "b" * 40 + "\n"),          # rev-parse -> mismatch
            ]
            with mock.patch.object(launch.subprocess, "run",
                                   side_effect=responses) as fake:
                with self.assertRaises(RuntimeError) as caught:
                    launch.fetch_pinned_source(Path(directory), sha)
            self.assertIn("not the pinned", str(caught.exception))
            self.assertEqual(fake.call_count, 5)

    def test_a_matching_head_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = "c" * 40
            responses = [
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, sha + "\n"),
            ]
            with mock.patch.object(launch.subprocess, "run", side_effect=responses):
                head = launch.fetch_pinned_source(Path(directory), sha)
            self.assertEqual(head, sha)

    def test_the_fetch_is_by_sha_and_not_by_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = "d" * 40
            responses = [
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, ""),
                FakeCompleted(0, sha + "\n"),
            ]
            with mock.patch.object(launch.subprocess, "run",
                                   side_effect=responses) as fake:
                launch.fetch_pinned_source(Path(directory), sha)
            fetch_call = fake.call_args_list[2]
            self.assertIn(sha, fetch_call.args[0])
            self.assertNotIn("--branch", fetch_call.args[0])


class ChildFailureTests(unittest.TestCase):
    def test_a_failed_training_child_is_a_failure_of_the_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            environment = {
                "COLAB_RELEASE_TAG": "test-release",
                "HALA_FREE_COLAB_RESEARCH": "1",
                "HALA_NO_PAID_RUNTIME_CONFIRMED": "1",
                "HALA_PINNED_SOURCE_SHA": "e" * 40,
                "H62_WORK": str(work),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch.object(Path, "exists", return_value=True):
                    with mock.patch.object(launch, "fetch_pinned_source",
                                           return_value="e" * 40):
                        with mock.patch.object(launch, "package_versions",
                                               return_value={"torch": "2.11.0",
                                                             "transformers": "5.16.1",
                                                             "peft": "0.20.0",
                                                             "PIL": "11.3.0"}):
                            with mock.patch.object(launch, "ensure_packages",
                                                   return_value={}):
                                with mock.patch.object(
                                    launch.subprocess, "run",
                                    return_value=FakeCompleted(137, "", "killed")):
                                    with self.assertRaises(RuntimeError) as caught:
                                        launch.main()
            self.assertIn("137", str(caught.exception))
            receipt = json.loads(
                (work / "h62-remote-launch-receipt.json").read_text(encoding="utf8"))
            self.assertIn("failure", receipt)
            self.assertEqual(receipt["colab_run_exit"], 137)
            self.assertNotEqual(receipt.get("status"), "completed")


if __name__ == "__main__":
    unittest.main()

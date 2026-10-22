"""H62 tests: a failed remote training/evaluation child MUST fail the parent.

All invocations are mocks; no Colab allocation, original dataset download,
GPU optimizer, checkpoint upload or provider API are executed.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from h61 import remote_entry as entry


class RemoteEntrypointFailClosedTests(TestCase):
    def test_successful_child_return_code_is_admissible(self):
        self.assertIsNone(entry.require_successful_child_exit(0))

    def test_child_failure_nonzero_or_invalid_is_rejected(self):
        for value in (1, 2, 137, -9, None, False, "0"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    entry.require_successful_child_exit(value)

    def test_must_pin_exact_remote_git_head(self):
        self.assertIsNone(entry.require_pinned_checkout(
            "5838f0b757693a7622dc6a40818be44928eec3b3",
            "5838f0b757693a7622dc6a40818be44928eec3b3"))
        for candidate in (None, "", "abc", "0"*40):
            with self.subTest(candidate=candidate):
                with self.assertRaises(RuntimeError):
                    entry.require_pinned_checkout(
                        "5838f0b757693a7622dc6a40818be44928eec3b3", candidate)

    def _execute_mocked(self, code):
        with TemporaryDirectory() as temp:
            real_exists = Path.exists
            def fake_exists(p):
                if str(p).replace("\\", "/").endswith("/content"):
                    return True
                return real_exists(p)
            versions={"torch":"2.11.0","transformers":"5.16.1",
                      "peft":"0.20.0","PIL":"11.3.0"}
            with (patch.object(Path, "exists", fake_exists),
                  patch.object(entry, "_git_available", return_value=True),
                  patch.object(entry, "package_versions", return_value=versions),
                  patch.object(entry, "run", return_value=SimpleNamespace(stdout="5838f0b757693a7622dc6a40818be44928eec3b3\n")),
                  patch.object(entry.subprocess, "run", return_value=SimpleNamespace(returncode=code)),
                  patch.dict(os.environ, {"H61_WORK":temp,"COLAB_RELEASE_TAG":"synthetic-unit-test",
                        "HALA_PINNED_SOURCE_SHA":"5838f0b757693a7622dc6a40818be44928eec3b3",
                        "HALA_REMOTE_SOURCE_BRANCH":"ps/h61-complete-corpus-and-curriculum"}, clear=False)):
                if code == 0:
                    self.assertEqual(entry.main(), 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, "child failed"):
                        entry.main()
            result=json.loads((Path(temp)/"remote-entry-receipt.json").read_text("utf8"))
            self.assertEqual(result["colab_run_exit"],code)
            self.assertEqual(result["steps"],["export","firstparty","curriculum",
                                               "baseline","official_base","train",
                                               "evaluate","official_adapter"])
            if code:
                self.assertIn("failure",result)
                self.assertIn(str(code),result["failure"])
            else:
                self.assertNotIn("failure",result)
            # A mocked 0 is not evidence that any real GPU was allocated.
            self.assertNotIn("neural_training_completed",result)

    def test_real_parent_main_aborts_on_mocked_child_exit_137(self):
        self._execute_mocked(137)

    def test_real_parent_main_does_not_abort_on_mocked_child_exit_zero(self):
        self._execute_mocked(0)

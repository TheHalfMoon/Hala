"""H64 regression tests: real subprocess contracts, receipts and watcher restarts."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from contextlib import redirect_stdout
import io
import random

from h62 import run_h61_diagnostic as runner
from h62 import colab_t4_watcher as watcher


PIN = "e" * 40
STEPS = "export,firstparty,curriculum,baseline,train,evaluate"


def launch_record(*, good=True):
    return {
        "status": "completed" if good else "failed",
        "colab_run_exit": 0,
        "repo_head": PIN,
        "pinned_source_sha": PIN,
        "repo_source_sha_verified": True,
    }


def run_record(*, good=True):
    return {
        "status": "completed" if good else "failed",
        "steps": STEPS.split(","),
        "failures": [],
    }


class SubprocessContractTests(unittest.TestCase):
    def test_wrapper_converts_genuine_completed_process_to_three_values(self):
        cp = subprocess.CompletedProcess(["colab", "sessions"], 0, "ok", "")
        with mock.patch.object(runner.subprocess, "run", return_value=cp):
            self.assertEqual(runner.run(["colab", "sessions"]), (0, "ok", ""))

    def test_session_hardware_parses_real_provider_output(self):
        stdout = "[hala-h62] node123 | Hardware: GPU | Shape: Standard | Variant: T4 | Status: IDLE"
        cp = subprocess.CompletedProcess(["colab", "sessions"], 0, stdout, "")
        with mock.patch.object(runner.subprocess, "run", return_value=cp):
            self.assertEqual(runner.session_hardware("hala-h62"), "GPU")

    def test_no_gpu_cannot_be_misreported_as_gpu(self):
        stdout = "[hala-h62] node123 | Hardware: CPU | Shape: Standard"
        cp = subprocess.CompletedProcess(["colab", "sessions"], 0, stdout, "")
        with mock.patch.object(runner.subprocess, "run", return_value=cp):
            self.assertEqual(runner.session_hardware("hala-h62"), "CPU")

    def test_execute_consumes_true_completed_process(self):
        cp = subprocess.CompletedProcess(["colab", "exec"], 137, "", "killed")
        with mock.patch.object(runner.subprocess, "run", return_value=cp):
            result = runner.execute("hala-h62", PIN, STEPS, "slider=1",
                                    "1", "1", "{}", 60)
        self.assertEqual(result, (137, "", "killed"))

    def test_download_does_not_claim_a_missing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            cp = subprocess.CompletedProcess(["colab", "download"], 0, "success", "")
            with mock.patch.object(runner.subprocess, "run", return_value=cp):
                self.assertFalse(runner.download("hala-h62", "/abs/remote", Path(folder) / "absent.json"))


class ReceiptAdmissionTests(unittest.TestCase):
    def invoke(self, *, launch=None, result=None, fail_download=None):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "launch-receipt.json").write_text(
                '{"status":"completed","pinned_source_sha":"stale"}', encoding="utf8"
            )
            launch = launch if launch is not None else launch_record()
            result = result if result is not None else run_record()

            def fake_download(_session, remote, destination):
                if fail_download and fail_download in str(remote):
                    return False
                payload = launch if "h62-remote-launch" in remote else result
                destination.write_text(json.dumps(payload), encoding="utf8")
                return True

            with (
                mock.patch.object(runner, "session_hardware", return_value="GPU"),
                mock.patch.object(runner, "execute", return_value=(0, "remote complete", "")),
                mock.patch.object(runner, "download", side_effect=fake_download),
            ):
                status = runner.main(["--pinned-sha", PIN, "--local-dir", folder,
                                      "--steps", STEPS])
            report = json.loads((directory / "run-report.json").read_text(encoding="utf8"))
            return status, report

    def test_all_verified_receipts_admit_completion(self):
        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertTrue(report["remote_receipts_verified"])
        self.assertEqual(report["status"], "completed")

    def test_successful_exec_without_download_is_still_failure(self):
        code, report = self.invoke(fail_download="run-receipt")
        self.assertEqual(code, 1)
        self.assertNotEqual(report.get("status"), "completed")

    def test_remote_launch_failure_blocks_completion(self):
        code, report = self.invoke(launch=launch_record(good=False))
        self.assertEqual(code, 1)
        self.assertIn("failure", report)

    def test_pinned_source_mismatch_blocks_completion(self):
        bad = launch_record()
        bad["repo_head"] = "f" * 40
        code, _ = self.invoke(launch=bad)
        self.assertEqual(code, 1)

    def test_undocumented_child_failure_blocks_completion(self):
        bad = launch_record()
        bad["colab_run_exit"] = 137
        code, _ = self.invoke(launch=bad)
        self.assertEqual(code, 1)

    def test_run_receipt_with_failed_step_blocks_completion(self):
        bad = run_record()
        bad["failures"] = ["GPU OOM"]
        code, _ = self.invoke(result=bad)
        self.assertEqual(code, 1)

    def test_run_receipt_with_wrong_steps_blocks_completion(self):
        bad = run_record()
        bad["steps"] = ["report"]
        code, _ = self.invoke(result=bad)
        self.assertEqual(code, 1)


class WatcherDurabilityTests(unittest.TestCase):
    def test_probe_only_consumes_the_three_field_provider_session_tuple(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(watcher.ProviderClient, "balance_is_zero",
                                   return_value=(True, "balance zero")):
                with mock.patch.object(watcher.ProviderClient, "existing_session",
                                       return_value=("hala-h62", "GPU", "")):
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        status = watcher.main(["--state-dir", folder, "--probe-only"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stream.getvalue())["live_hardware"], "GPU")

    def test_original_deadline_is_restored_from_first_start_record(self):
        with tempfile.TemporaryDirectory() as folder:
            log = watcher.ReceiptLog(Path(folder) / "receipts.jsonl")
            log.append({"event": "start", "state": {"started_utc": "2026-10-22T00:00:00Z"}})
            log.append({"event": "attempt", "attempt": 1, "request": 1, "outcome":
                        watcher.CAPACITY_OR_BACKEND_UNAVAILABLE})
            state = log.restore()
            self.assertEqual(state.started_utc, "2026-10-22T00:00:00Z")
            self.assertEqual(state.requests, 1)

    def test_expired_original_budget_refuses_provider_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            log = watcher.ReceiptLog(Path(folder) / "receipts.jsonl")
            fake = mock.Mock()
            state = watcher.run_watcher(
                fake, log, watcher.WatcherState(), watcher.BackoffPolicy(),
                deadline_seconds=60, elapsed_seconds=61, max_requests=10,
                rng=random.Random(1), clock=lambda: 0.0)
            self.assertEqual(state.stop_reason, "deadline_expired")
            fake.balance_is_zero.assert_not_called()

    def test_stopped_watcher_does_not_restart_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            log = watcher.ReceiptLog(Path(folder) / "receipts.jsonl")
            log.append({"event": "stop", "reason": "request_budget_exhausted"})
            state = log.restore()
            fake = mock.Mock()
            restored = watcher.run_watcher(
                fake, log, state, watcher.BackoffPolicy(),
                deadline_seconds=1000, max_requests=80,
                rng=random.Random(1), clock=lambda: 0.0)
            self.assertEqual(restored.stop_reason, "request_budget_exhausted")
            fake.balance_is_zero.assert_not_called()

    def test_allocated_watcher_does_not_launch_again_on_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            log = watcher.ReceiptLog(Path(folder) / "receipts.jsonl")
            log.append({"event": "attempt", "attempt": 1, "request": 1,
                        "outcome": watcher.ALLOCATED, "session": "hala-h62"})
            fake = mock.Mock()
            launch = mock.Mock()
            state = watcher.run_watcher(
                fake, log, log.restore(), watcher.BackoffPolicy(),
                deadline_seconds=500, max_requests=80,
                rng=random.Random(1), clock=lambda: 0.0, launch=launch)
            self.assertTrue(state.allocated)
            launch.assert_not_called()
            fake.balance_is_zero.assert_not_called()

    def test_invalid_elapsed_time_is_refused_before_allocate(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = mock.Mock()
            with self.assertRaises(ValueError):
                watcher.run_watcher(
                    fake, watcher.ReceiptLog(Path(folder) / "receipts.jsonl"),
                    watcher.WatcherState(), watcher.BackoffPolicy(),
                    deadline_seconds=60, elapsed_seconds=-2,
                    max_requests=10, rng=random.Random(1), clock=lambda: 0.0)
            fake.balance_is_zero.assert_not_called()


if __name__ == "__main__":
    unittest.main()

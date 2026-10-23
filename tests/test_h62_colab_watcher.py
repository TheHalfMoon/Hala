"""H62 the durable free-T4 watcher: the behaviour that must hold on the wire.

These tests never contact the provider. They drive the watcher through a
fake command runner that replays the provider answers that were actually
observed on this account, and they assert the properties that make the
mechanism safe to leave running unattended.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from h62.colab_t4_watcher import (
    ALLOCATED,
    AUTHENTICATION_FAILED,
    CAPACITY_OR_BACKEND_UNAVAILABLE,
    ENTITLEMENT_BLOCKED,
    QUOTA_BLOCKED,
    ReceiptLog,
    BackoffPolicy,
    ProviderClient,
    WatcherState,
    classify_provider_error,
    parse_retry_after,
    redact,
    run_watcher,
)

# The exact provider text observed on this account when a free T4 was asked
# for and the answer was a 503. The CPU variant of the same request on the
# same account and the same CLI succeeded.
OBSERVED_T4_503 = (
    "ColabRequestError: Failed to issue request POST "
    "https://colab.research.google.com/tun/m/assign?nbh=1e713701&variant=GPU"
    "&accelerator=T4: Service Unavailable"
)
OBSERVED_T4_READY = "[colab] Creating session 'hala-h62'...\n[colab] Session READY."
OBSERVED_CPU_READY = "[colab] Creating session 'hala-h62-cpu'...\n[colab] Session READY."
OBSERVED_PRECONDITION = "Error: precondition failed"
OBSERVED_USAGE_ZERO = "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"


class FakeRunner:
    """A scripted provider that answers by COMMAND, like the real CLI.

    The provider CLI is not a fixed sequence of replies: it is a set of
    commands, each of which answers from the current provider state. The
    fake models that, so the tests exercise the same command flow the
    watcher uses against the real client.
    """

    def __init__(self, usage=None, sessions=None, allocate=None):
        self.usage = usage if usage is not None else (0, OBSERVED_USAGE_ZERO, "")
        # `sessions` may be a single reply or a list consumed in order, with
        # the last entry repeating.
        self.sessions = sessions if sessions is not None else (0, "[colab] No active sessions found on server.", "")
        if isinstance(self.sessions, list):
            self._session_queue = list(self.sessions)
        else:
            self._session_queue = []
        self.allocate = allocate if allocate is not None else (1, "", OBSERVED_T4_503)
        self.calls = []
        self.allocations = 0

    def run(self, arguments):
        self.calls.append(list(arguments))
        command = arguments[0]
        if command == "usage":
            return self.usage
        if command == "sessions":
            if self._session_queue:
                return self._session_queue.pop(0) if len(self._session_queue) > 1 else self._session_queue[0]
            return self.sessions
        if command == "new":
            self.allocations += 1
            return self.allocate
        return (1, "", f"unexpected command {command}")


def usage_zero():
    return (0, OBSERVED_USAGE_ZERO, "")


def session_line(name, hardware, status="IDLE"):
    endpoint = "m-s-kkb-usw1b1-y6688rpg69hj"
    return (0, f"[{name}] {endpoint} | Hardware: {hardware} | Shape: Standard "
               f"| Variant: DEFAULT | Status: {status}", "")


class OutcomeClassificationTests(unittest.TestCase):
    def test_the_observed_503_is_a_capacity_condition(self):
        self.assertEqual(classify_provider_error(OBSERVED_T4_503),
                         CAPACITY_OR_BACKEND_UNAVAILABLE)

    def test_a_503_is_not_reported_as_an_entitlement_block(self):
        # The provider does not distinguish the causes of a 503. Recording it
        # as a capacity condition is the honest reading; recording it as an
        # entitlement block would be a claim the evidence does not support.
        outcome = classify_provider_error(OBSERVED_T4_503)
        self.assertNotEqual(outcome, ENTITLEMENT_BLOCKED)
        self.assertNotEqual(outcome, QUOTA_BLOCKED)

    def test_entitlement_quota_and_auth_are_distinguished(self):
        self.assertEqual(classify_provider_error(OBSERVED_PRECONDITION), ENTITLEMENT_BLOCKED)
        self.assertEqual(classify_provider_error("HTTP 403 Forbidden"), ENTITLEMENT_BLOCKED)
        self.assertEqual(classify_provider_error("HTTP 429 Too Many Requests"), QUOTA_BLOCKED)
        self.assertEqual(classify_provider_error("HTTP 401 Unauthorized"), AUTHENTICATION_FAILED)

    def test_an_unrecognised_answer_is_not_guessed(self):
        self.assertEqual(classify_provider_error("something else entirely"), "UNKNOWN_FAILURE")
        self.assertEqual(classify_provider_error(""), "UNKNOWN_FAILURE")

    def test_retry_after_is_read_in_seconds(self):
        self.assertEqual(parse_retry_after("Retry-After: 45"), 45.0)
        self.assertEqual(parse_retry_after("retry after 90 seconds"), 90.0)
        self.assertEqual(parse_retry_after("no hint here"), 0.0)

    def test_secrets_are_redacted_from_the_audit_trail(self):
        text = "authorization: Bearer abc123def token=shhh password=hunter2"
        cleaned = redact(text)
        self.assertNotIn("abc123def", cleaned)
        self.assertNotIn("hunter2", cleaned)


class BackoffPolicyTests(unittest.TestCase):
    def test_the_wait_grows_and_stays_within_the_cap(self):
        policy = BackoffPolicy(initial=120.0, factor=2.0, maximum=1800.0, jitter=0.25)
        rng = random.Random(7)
        previous_base = 0.0
        for attempt in range(1, 12):
            wait = policy.delay(attempt, rng)
            base = min(1800.0, 120.0 * (2.0 ** (attempt - 1)))
            self.assertGreaterEqual(wait, 1.0)
            self.assertLessEqual(wait, 1800.0)
            # The jitter is symmetric and bounded, so the wait stays inside
            # the band around the base delay and never leaves the cap.
            self.assertGreaterEqual(wait, base * 0.75 - 1e-9)
            self.assertLessEqual(wait, base * 1.25 + 1e-9)
            # The BASE delay is monotonic. The jittered wait deliberately is
            # not, so that the request pattern does not look like a metronome.
            self.assertGreaterEqual(base, previous_base)
            previous_base = base

    def test_the_jitter_actually_varies_the_wait(self):
        policy = BackoffPolicy(initial=120.0, factor=2.0, maximum=1800.0, jitter=0.25)
        waits = {policy.delay(4, random.Random(seed)) for seed in range(20)}
        self.assertGreater(len(waits), 1)

    def test_an_invalid_attempt_number_is_rejected(self):
        with self.assertRaises(ValueError):
            BackoffPolicy().delay(0, random.Random(1))


class ReceiptDurabilityTests(unittest.TestCase):
    def test_a_restarted_watcher_continues_instead_of_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.jsonl"
            log = ReceiptLog(path)
            log.append({"event": "attempt", "attempt": 3, "request": 3,
                        "outcome": CAPACITY_OR_BACKEND_UNAVAILABLE, "error": "503"})
            log.append({"event": "attempt", "attempt": 4, "request": 4,
                        "outcome": ALLOCATED, "session": "hala-h62"})
            restored = log.restore()
            self.assertEqual(restored.attempts, 4)
            self.assertEqual(restored.requests, 4)
            self.assertTrue(restored.allocated)
            self.assertEqual(restored.session_name, "hala-h62")

    def test_every_receipt_is_a_complete_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.jsonl"
            log = ReceiptLog(path)
            for index in range(5):
                log.append({"event": "attempt", "attempt": index})
            lines = path.read_text(encoding="utf8").splitlines()
            self.assertEqual(len(lines), 5)
            for line in lines:
                self.assertIn("recorded_utc", json.loads(line))


class WatcherBehaviourTests(unittest.TestCase):
    def _client(self, **kwargs):
        return ProviderClient(FakeRunner(**kwargs), "hala-h62")

    def test_the_paid_compute_guard_refuses_to_allocate(self):
        # A non-zero balance means the account is spending. The watcher stops.
        runner = FakeRunner(usage=(0, "Current balance: 12.50 compute units", ""))
        client = ProviderClient(runner, "hala-h62")
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=60.0, max_requests=10,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock())
            self.assertFalse(state.allocated)
            self.assertEqual(state.stop_reason, "paid_compute_guard")
            self.assertEqual(runner.allocations, 0)

    def test_an_existing_gpu_session_is_adopted_without_a_second_request(self):
        runner = FakeRunner(sessions=session_line("hala-h62", "T4"))
        client = ProviderClient(runner, "hala-h62")
        launched = []
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=60.0, max_requests=10,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock(),
                                launch=lambda name, adopted: launched.append((name, adopted)))
            self.assertTrue(state.allocated)
            self.assertTrue(state.adopted)
            self.assertEqual(launched, [("hala-h62", True)])
            self.assertEqual(runner.allocations, 0)

    def test_a_cpu_session_does_not_block_the_t4_request(self):
        # The CPU assignment is evidence that the assignment path works. It
        # must not be mistaken for a GPU.
        runner = FakeRunner(
            sessions=[session_line("hala-h62-cpu", "CPU"),
                      (0, "[colab] No active sessions found on server.", "")],
            allocate=(0, OBSERVED_T4_READY, ""),
        )
        client = ProviderClient(runner, "hala-h62")
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=60.0, max_requests=10,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock())
            self.assertTrue(state.allocated)
            self.assertFalse(state.adopted)
            self.assertEqual(runner.allocations, 1)

    def test_a_503_keeps_the_watcher_running_and_the_launch_is_not_called(self):
        runner = FakeRunner(allocate=(1, "", OBSERVED_T4_503))
        client = ProviderClient(runner, "hala-h62")
        launched = []
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=600.0, max_requests=3,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock(),
                                launch=lambda name, adopted: launched.append(name))
            self.assertFalse(state.allocated)
            self.assertEqual(launched, [])
            self.assertEqual(state.requests, 3)
            self.assertEqual(state.outcome, CAPACITY_OR_BACKEND_UNAVAILABLE)

    def test_a_terminal_entitlement_block_stops_the_watcher(self):
        runner = FakeRunner(allocate=(1, "", OBSERVED_PRECONDITION))
        client = ProviderClient(runner, "hala-h62")
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=600.0, max_requests=10,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock())
            self.assertFalse(state.allocated)
            self.assertIn("ENTITLEMENT_BLOCKED", state.stop_reason)
            self.assertEqual(runner.allocations, 1)

    def test_the_request_budget_is_never_exceeded(self):
        runner = FakeRunner(allocate=(1, "", OBSERVED_T4_503))
        client = ProviderClient(runner, "hala-h62")
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=10 ** 6, max_requests=5,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock())
            self.assertFalse(state.allocated)
            self.assertEqual(state.stop_reason, "request_budget_exhausted")
            self.assertEqual(runner.allocations, 5)

    def test_the_deadline_stops_the_watcher(self):
        runner = FakeRunner(allocate=(1, "", OBSERVED_T4_503))
        client = ProviderClient(runner, "hala-h62")
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            clock = _FakeClock(start=0.0, step=400.0)
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=500.0, max_requests=1000,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=clock)
            self.assertFalse(state.allocated)
            self.assertEqual(state.stop_reason, "deadline_expired")

    def test_the_launch_runs_exactly_once_on_allocation(self):
        runner = FakeRunner(allocate=(0, OBSERVED_T4_READY, ""))
        client = ProviderClient(runner, "hala-h62")
        launched = []
        with tempfile.TemporaryDirectory() as directory:
            log = ReceiptLog(Path(directory) / "r.jsonl")
            state = run_watcher(client, log, WatcherState(), BackoffPolicy(),
                                deadline_seconds=60.0, max_requests=10,
                                rng=random.Random(1), sleep=lambda _: None,
                                clock=_FakeClock(),
                                launch=lambda name, adopted: launched.append(name))
            self.assertTrue(state.allocated)
            self.assertEqual(launched, ["hala-h62"])
            records = log.read()
            allocations = [r for r in records if r.get("outcome") == ALLOCATED]
            self.assertEqual(len(allocations), 1)


class _FakeClock:
    """A monotonic clock that advances a fixed amount on every reading."""

    def __init__(self, start=0.0, step=0.0):
        self.now = start
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


if __name__ == "__main__":
    unittest.main()

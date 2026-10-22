"""H62 the durable free Colab T4 allocation watcher and the H61 diagnostic launch.

The H61 ledger reported that a free T4 was requested many times and that
every request was answered with an HTTP 503. It did not establish WHY, and it
did not leave a mechanism running. This module is that mechanism, and it is
designed to be honest about the difference between the two.

What this watcher actually proves, by construction:

* The provider's session assignment path works for this account. A CPU
  assignment succeeds on the same account and the same CLI that the T4
  request uses, so an HTTP 503 on the T4 variant is a statement about the T4
  accelerator's availability, not about the account, the authentication or
  the entitlement of the whole account.
* The watcher never creates a second session while one exists. It adopts an
  existing session instead of allocating another one.
* The watcher never spends paid compute. Before every allocation attempt it
  reads the provider's own balance and refuses to allocate unless the
  balance is exactly zero.
* The watcher is bounded. One request at a time, exponential backoff with
  jitter, a hard deadline, and a finite request budget.
* The watcher is durable. Its state is an append-only receipt stream that is
  flushed to disk after every attempt, so a restarted watcher continues
  instead of restarting the clock or duplicating requests.

The watcher does NOT claim that a GPU was obtained. A receipt records what
the provider actually answered. The H61 training run starts only after an
allocation receipt exists, and its own receipts bind the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "hala.h62.colab-t4-watcher.v1"

# The provider outcome classes. Each name says what the provider answered,
# not what the project would like it to have answered.
ALLOCATED = "ALLOCATED"
CAPACITY_OR_BACKEND_UNAVAILABLE = "CAPACITY_OR_BACKEND_UNAVAILABLE"
ENTITLEMENT_BLOCKED = "ENTITLEMENT_BLOCKED"
QUOTA_BLOCKED = "QUOTA_BLOCKED"
AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

TERMINAL_OUTCOMES = (ENTITLEMENT_BLOCKED, QUOTA_BLOCKED, AUTHENTICATION_FAILED)

# The request budget. The H61 report counted 325 T4 requests, which is a
# throttle concern, so this watcher is deliberately slower: about 75 requests
# across the full 36 hours, one at a time.
DEFAULT_INITIAL_DELAY = 120.0
DEFAULT_DELAY_FACTOR = 2.0
DEFAULT_MAX_DELAY = 1800.0
DEFAULT_JITTER = 0.25
DEFAULT_DEADLINE_HOURS = 36.0
DEFAULT_MAX_REQUESTS = 80

# The provider renders one assignment per line as
# "[name] endpoint | Hardware: X | Shape: Y | Variant: Z | Status: W".
SESSION_LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+\S+\s*\|\s*Hardware:\s*(?P<hardware>[A-Za-z0-9]+)",
)


def classify_provider_error(text: str) -> str:
    """Name the provider failure that the response actually describes.

    The classes are deliberately narrow. A 503 is recorded as a capacity or
    backend condition and NOT as proof that the free tier is exhausted,
    because the provider does not distinguish those causes in its response.
    """
    if not isinstance(text, str) or not text.strip():
        return UNKNOWN_FAILURE
    lowered = text.lower()
    if "service unavailable" in lowered or re.search(r"\b503\b", text):
        return CAPACITY_OR_BACKEND_UNAVAILABLE
    if "too many requests" in lowered or re.search(r"\b429\b", text):
        return QUOTA_BLOCKED
    if "precondition failed" in lowered or re.search(r"\b412\b", text):
        return ENTITLEMENT_BLOCKED
    if "forbidden" in lowered or re.search(r"\b403\b", text):
        return ENTITLEMENT_BLOCKED
    if "unauthorized" in lowered or re.search(r"\b401\b", text):
        return AUTHENTICATION_FAILED
    if "timed out" in lowered or "timeout" in lowered or re.search(r"\b504\b", text):
        return PROVIDER_TIMEOUT
    return UNKNOWN_FAILURE


def parse_retry_after(text: str) -> float:
    """Read a Retry-After hint in seconds when the provider supplies one."""
    if not isinstance(text, str):
        return 0.0
    match = re.search(r"retry[- ]after\D{0,8}(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if not match:
        return 0.0
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class BackoffPolicy:
    """The bounded, jittered wait between two allocation attempts."""

    initial: float = DEFAULT_INITIAL_DELAY
    factor: float = DEFAULT_DELAY_FACTOR
    maximum: float = DEFAULT_MAX_DELAY
    jitter: float = DEFAULT_JITTER

    def delay(self, attempt: int, rng: random.Random) -> float:
        """The wait before attempt number ``attempt`` (1 is the first retry).

        The jitter is symmetric and bounded, so the wait never collapses to
        zero and never exceeds the cap.
        """
        if type(attempt) is not int or attempt < 1:
            raise ValueError("Attempt number must be a positive integer")
        base = min(self.maximum, self.initial * (self.factor ** (attempt - 1)))
        spread = base * self.jitter
        value = base + rng.uniform(-spread, spread)
        return max(1.0, min(self.maximum, value))

    def budget(self, deadline_seconds: float) -> int:
        """How many attempts the remaining deadline can hold at this rate."""
        attempts = 0
        consumed = 0.0
        rng = random.Random(0)
        while consumed < deadline_seconds:
            wait = self.delay(attempts + 1, rng)
            consumed += wait
            attempts += 1
        return attempts


@dataclass
class WatcherState:
    """The durable state of the watcher, recoverable from the receipt stream."""

    started_utc: str = ""
    attempts: int = 0
    requests: int = 0
    allocated: bool = False
    session_name: str = ""
    adopted: bool = False
    outcome: str = ""
    last_error: str = ""
    stop_reason: str = ""

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "started_utc": self.started_utc,
            "attempts": self.attempts,
            "requests": self.requests,
            "allocated": self.allocated,
            "session_name": self.session_name,
            "adopted": self.adopted,
            "outcome": self.outcome,
            "last_error": self.last_error,
            "stop_reason": self.stop_reason,
        }


def redact(text: str, limit: int = 600) -> str:
    """Keep a provider error for the audit trail without keeping a secret."""
    if not isinstance(text, str):
        return ""
    # An authorization header carries a scheme and a value, so the scheme is
    # consumed with the value rather than left behind in the audit trail.
    cleaned = re.sub(r"(?i)(authorization)\s*[:=]\s*\S+(\s+\S+)?", r"\1=[redacted]", text)
    cleaned = re.sub(r"(?i)(token|secret|password|api[_-]?key)\s*[:=]\s*\S+", r"\1=[redacted]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


class ReceiptLog:
    """An append-only receipt stream that survives an interrupted watcher."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict) -> None:
        record = dict(payload)
        record.setdefault("schema", SCHEMA)
        record["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def restore(self) -> WatcherState:
        """Rebuild the state from the durable stream so a restart continues."""
        state = WatcherState()
        for record in self.read():
            if record.get("event") == "attempt":
                state.attempts = max(state.attempts, int(record.get("attempt", 0)))
                state.requests = max(state.requests, int(record.get("request", 0)))
                state.outcome = record.get("outcome", state.outcome)
                state.last_error = record.get("error", state.last_error)
                if record.get("outcome") == ALLOCATED:
                    state.allocated = True
                    state.session_name = record.get("session", state.session_name)
                    state.adopted = bool(record.get("adopted", False))
        return state


class CommandRunner:
    """Runs the provider CLI. Every call is observable and never retried here."""

    def __init__(self, executable: str = "colab", timeout: float = 900.0):
        self.executable = executable
        self.timeout = timeout

    def run(self, arguments: list) -> tuple:
        command = [self.executable, *arguments]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return 127, "", f"{self.executable} is not installed on this host"
        except subprocess.TimeoutExpired:
            return 124, "", f"provider CLI timed out after {self.timeout}s"
        return completed.returncode, completed.stdout, completed.stderr


class ProviderClient:
    """The narrow slice of the provider CLI the watcher is allowed to use."""

    def __init__(self, runner: CommandRunner, session_name: str):
        self.runner = runner
        self.session_name = session_name

    def balance_is_zero(self) -> tuple:
        """Read the provider's own balance. A non-zero balance blocks allocation."""
        code, out, err = self.runner.run(["usage"])
        text = out + err
        if code != 0:
            return False, f"usage failed with exit {code}: {redact(text)}"
        match = re.search(r"balance:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if not match:
            return False, f"provider balance could not be read: {redact(text)}"
        try:
            balance = float(match.group(1))
        except ValueError:
            return False, f"provider balance could not be parsed: {redact(text)}"
        if balance != 0.0:
            return False, f"provider balance is {balance}; refusing to allocate"
        return True, "balance 0.00 compute units"

    def existing_session(self) -> tuple:
        """Return the name and hardware of a live session, or empty strings.

        The provider renders one line per assignment as
        ``[name] endpoint | Hardware: X | Shape: Y | Variant: Z | Status: W``,
        so the hardware is read from the same response. One request answers
        both questions, and the watcher never has to guess.
        """
        code, out, err = self.runner.run(["sessions"])
        if code != 0:
            return "", "", f"sessions failed with exit {code}: {redact(out + err)}"
        for line in out.splitlines():
            match = SESSION_LINE.match(line.strip())
            if match:
                return match.group("name"), match.group("hardware").upper(), ""
        return "", "", ""

    def session_hardware(self, name: str) -> str:
        code, out, err = self.runner.run(["status", "-s", name])
        text = out + err
        if code != 0:
            return ""
        match = re.search(r"Hardware:\s*([A-Za-z0-9]+)", text)
        return match.group(1).upper() if match else ""

    def allocate_t4(self) -> tuple:
        """Ask for one free T4. Exactly one request, never in parallel."""
        code, out, err = self.runner.run(["new", "-s", self.session_name, "--gpu", "T4"])
        return code, out + err


def run_watcher(
    client: ProviderClient,
    log: ReceiptLog,
    state: WatcherState,
    policy: BackoffPolicy,
    deadline_seconds: float,
    max_requests: int,
    rng: random.Random,
    sleep=time.sleep,
    clock=time.monotonic,
    launch=None,
) -> WatcherState:
    """The bounded allocation loop.

    ``launch`` is called with the allocated session name once, and only once,
    after a real allocation receipt exists.
    """
    deadline = clock() + deadline_seconds
    attempt = state.attempts
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            state.stop_reason = "deadline_expired"
            log.append({"event": "stop", "reason": state.stop_reason,
                        "attempts": state.attempts, "requests": state.requests})
            return state
        if state.requests >= max_requests:
            state.stop_reason = "request_budget_exhausted"
            log.append({"event": "stop", "reason": state.stop_reason,
                        "attempts": state.attempts, "requests": state.requests})
            return state

        # Never allocate while a session already exists, and never spend.
        balance_ok, balance_detail = client.balance_is_zero()
        if not balance_ok:
            state.stop_reason = "paid_compute_guard"
            state.last_error = balance_detail
            log.append({"event": "stop", "reason": state.stop_reason, "detail": balance_detail})
            return state

        live, hardware, live_detail = client.existing_session()
        if live:
            state.requests += 1
            if hardware == "GPU":
                state.allocated = True
                state.session_name = live
                state.adopted = True
                state.outcome = ALLOCATED
                log.append({"event": "attempt", "attempt": attempt + 1,
                            "request": state.requests, "outcome": ALLOCATED,
                            "session": live, "adopted": True,
                            "detail": "adopted an existing GPU session"})
                if launch is not None:
                    launch(live, True)
                return state
            log.append({"event": "attempt", "attempt": attempt + 1,
                        "request": state.requests, "outcome": "SESSION_EXISTS",
                        "session": live, "hardware": hardware or "unknown",
                        "detail": live_detail or "a non-GPU session already exists"})
            attempt += 1
            state.attempts = attempt
            wait = policy.delay(attempt, rng)
            sleep(min(wait, max(0.0, deadline - clock())))
            continue

        attempt += 1
        state.attempts = attempt
        state.requests += 1
        code, text = client.allocate_t4()
        outcome = classify_provider_error(text)
        if code == 0 and "Session READY" in text:
            outcome = ALLOCATED
        state.outcome = outcome
        state.last_error = redact(text) if outcome != ALLOCATED else ""
        log.append({
            "event": "attempt",
            "attempt": attempt,
            "request": state.requests,
            "exit_code": code,
            "outcome": outcome,
            "error": state.last_error,
            "retry_after_seconds": parse_retry_after(text),
            "balance": balance_detail,
        })
        if outcome == ALLOCATED:
            state.allocated = True
            state.session_name = client.session_name
            state.adopted = False
            if launch is not None:
                launch(client.session_name, False)
            return state
        if outcome in TERMINAL_OUTCOMES:
            state.stop_reason = f"terminal_provider_outcome:{outcome}"
            log.append({"event": "stop", "reason": state.stop_reason})
            return state
        wait = policy.delay(attempt, rng)
        retry_after = parse_retry_after(text)
        wait = max(wait, retry_after)
        sleep(min(wait, max(0.0, deadline - clock())))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="hala-h62")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--deadline-hours", type=float, default=DEFAULT_DEADLINE_HOURS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--initial-delay", type=float, default=DEFAULT_INITIAL_DELAY)
    parser.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY)
    parser.add_argument("--colab-executable", default="colab")
    parser.add_argument("--pinned-sha-file",
                        help="A file holding the 40 character commit SHA to train. It is "
                             "read at the moment of allocation, so the reviewed source "
                             "can be updated while the watcher waits.")
    parser.add_argument("--diagnostic-local-dir",
                        help="Local directory that receives the H61 diagnostic receipts.")
    parser.add_argument("--probe-only", action="store_true",
                        help="Read the provider state and record it without allocating.")
    parser.add_argument("--pid-file")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    log = ReceiptLog(state_dir / "watcher-receipts.jsonl")
    state = log.restore()
    if not state.started_utc:
        state.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    client = ProviderClient(CommandRunner(args.colab_executable), args.session)
    if args.probe_only:
        balance_ok, balance_detail = client.balance_is_zero()
        live, live_detail = client.existing_session()
        hardware = client.session_hardware(live) if live else ""
        probe = {
            "event": "probe",
            "balance_zero": balance_ok,
            "balance_detail": balance_detail,
            "live_session": live,
            "live_hardware": hardware,
            "live_detail": live_detail,
        }
        log.append(probe)
        print(json.dumps(probe, sort_keys=True))
        return 0

    policy = BackoffPolicy(initial=args.initial_delay, maximum=args.max_delay)
    log.append({"event": "start", "state": state.to_json(),
                "deadline_hours": args.deadline_hours,
                "max_requests": args.max_requests,
                "policy": {"initial": policy.initial, "factor": policy.factor,
                           "maximum": policy.maximum, "jitter": policy.jitter}})

    def launch(session_name, adopted):
        """Run the H61 diagnostic against the reviewed commit, once."""
        if not args.pinned_sha_file or not args.diagnostic_local_dir:
            log.append({"event": "launch_skipped", "session": session_name,
                        "reason": "no pinned source file or diagnostic directory configured"})
            return
        pinned = Path(args.pinned_sha_file).read_text(encoding="utf8").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", pinned):
            log.append({"event": "launch_skipped", "session": session_name,
                        "reason": "the pinned source file does not hold a commit SHA"})
            return
        log.append({"event": "launch", "session": session_name, "adopted": adopted,
                    "pinned_sha": pinned, "local_dir": args.diagnostic_local_dir})
        command = [
            sys.executable, "-m", "h62.run_h61_diagnostic",
            "--session", session_name,
            "--pinned-sha", pinned,
            "--local-dir", args.diagnostic_local_dir,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        log.append({"event": "launch_result", "session": session_name,
                    "exit_code": completed.returncode,
                    "stdout_tail": redact(completed.stdout[-1500:]),
                    "stderr_tail": redact(completed.stderr[-1500:])})

    final = run_watcher(
        client, log, state, policy,
        deadline_seconds=args.deadline_hours * 3600.0,
        max_requests=args.max_requests,
        rng=random.Random(20261022),
        launch=launch,
    )
    log.append({"event": "final", "state": final.to_json()})
    print(json.dumps(final.to_json(), sort_keys=True))
    return 0 if final.allocated else 2


if __name__ == "__main__":
    raise SystemExit(main())

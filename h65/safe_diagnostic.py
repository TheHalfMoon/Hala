"""H65 host-supervised Colab diagnostic, with continuous checkpoint egress.

Runs only on an already allocated T4. This does not allocate a session or
run paid compute. The legacy H62 runner executes in one bounded child; an
independent host-side loop attempts read-only SHA-verified checkpoint egress
while its remote work runs. Failures are recorded and cannot mean success.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from h65.checkpoint_egress import snapshot_once
from h62.run_h61_diagnostic import session_hardware

SCHEMA = "hala.h65.supervised-diagnostic.v1"


def write_receipt(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def supervise(session: str, sha: str, destination: Path, *,
              interval: float = 90.0, max_seconds: float = 21600.0,
              steps: str = "export,firstparty,curriculum,baseline,train,evaluate",
              runner=subprocess.Popen,
              snapshot=snapshot_once,
              hardware=session_hardware,
              sleep=time.sleep,
              clock=time.monotonic) -> dict:
    """Return truthful result; never create or adopt a GPU session."""
    if not 30.0 <= interval <= 1800.0 or not 30.0 <= max_seconds <= 28800.0:
        raise ValueError("bounded monitor interval and deadline required")
    if (len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha)):
        raise ValueError("40-character pinned commit required")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "monitor-receipts.jsonl"
    last_status = "NOT_YET_ATTEMPTED"
    successful_snapshots = 0
    absent = 0
    created = clock()
    exit_code = None
    if hardware(session) != "T4":
        raise RuntimeError("the dedicated T4 must be present before launch")
    command = [sys.executable, "-m", "h62.run_h61_diagnostic",
               "--session", session, "--pinned-sha", sha,
               "--local-dir", str(destination / "operator"),
               "--steps", steps, "--timeout", str(int(max_seconds))]
    log = destination / "operator.log"
    with log.open("w", encoding="utf8") as output:
        child = runner(command, stdout=output, stderr=subprocess.STDOUT)
        write_receipt(receipt_path, {"event": "launch", "session": session,
                                    "pinned_sha": sha, "steps": steps, "pid": child.pid})
        try:
            while child.poll() is None:
                try:
                    result = snapshot(session, destination / "checkpoint-staging")
                    last_status = result["status"]
                    if last_status in ("VERIFIED_HOST_STAGING", "ALREADY_VERIFIED"):
                        successful_snapshots += 1
                    write_receipt(receipt_path, {"event": "egress", "status": last_status,
                                                "step": result.get("step")})
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    last_status = "EGRESS_FAILED"
                    write_receipt(receipt_path, {"event": "egress_error",
                                                "error_class": type(error).__name__})
                # Colab can discard an interactive GPU without notifying the
                # executing CLI. Detect three consecutive missing sessions.
                try:
                    present = hardware(session) == "T4"
                except (OSError, ValueError, RuntimeError):
                    present = False
                absent = 0 if present else absent + 1
                write_receipt(receipt_path, {"event": "provider_probe",
                                            "present": present, "consecutive_missing": absent})
                if absent >= 3 or clock() - created >= max_seconds:
                    reason = "provider_session_lost" if absent >= 3 else "monitor_deadline"
                    write_receipt(receipt_path, {"event": "stop", "reason": reason})
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    exit_code = child.returncode
                    return {"schema": SCHEMA, "status": "FAILED",
                            "reason": reason, "child_exit": exit_code,
                            "successful_egress_polls": successful_snapshots}
                sleep(interval)
            exit_code = child.wait()
            # After completed execution the session may still exist, so take
            # one final bounded, validated snapshot before the operator exits.
            try:
                result = snapshot(session, destination / "checkpoint-staging")
                last_status = result["status"]
                if last_status in ("VERIFIED_HOST_STAGING", "ALREADY_VERIFIED"):
                    successful_snapshots += 1
            except (OSError, ValueError, RuntimeError, KeyError):
                last_status = "EGRESS_FAILED"
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
    if exit_code != 0:
        status, reason = "FAILED", "child_nonzero"
    elif "train" in [part.strip() for part in steps.split(",")] and not successful_snapshots:
        status, reason = "FAILED", "missing_verified_checkpoint_egress"
    else:
        status, reason = "COMPLETED_WITH_HOST_EGRESS", "host_evidence_only"
    report = {"schema": SCHEMA, "status": status, "reason": reason,
              "child_exit": exit_code, "successful_egress_polls": successful_snapshots,
              "last_egress": last_status,
              "private_provider_readback_verified": False}
    write_receipt(receipt_path, {"event": "final", **report})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pinned-sha", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=90.0)
    parser.add_argument("--max-seconds", type=float, default=21600.0)
    parser.add_argument("--steps",
                        default="export,firstparty,curriculum,baseline,train,evaluate")
    args = parser.parse_args(argv)
    result = supervise(args.session, args.pinned_sha, args.destination,
                       interval=args.interval, max_seconds=args.max_seconds,
                       steps=args.steps)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETED_WITH_HOST_EGRESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

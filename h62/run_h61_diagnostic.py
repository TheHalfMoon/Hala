"""H62 the local side of the H61 diagnostic: drive the remote session, keep the receipts.

This is the operator-side half of `h62.remote_h62_launch`. It verifies that
the session really has a GPU, runs the pinned remote entry point inside it,
keeps the complete provider output as a local log, and reads the receipts
and the checkpoint bundle back off the remote filesystem.

It fails closed in the same direction as the remote entry point: if the
remote child fails, the local runner reports the failure and does not
publish a success. The local log is kept whatever happens, because a
preserved failure is evidence and a discarded failure is not.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve().parent / "remote_h62_launch.py"
REMOTE_WORK = "/content/hala-h62-work"
SESSION_LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+\S+\s*\|\s*Hardware:\s*(?P<hardware>[A-Za-z0-9]+)")


def run(command, timeout=None):
    print(f"H62_LOCAL_RUN {command}", flush=True)
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def session_hardware(session: str) -> str:
    code, out, err = run(["colab", "sessions"])
    if code != 0:
        raise RuntimeError(f"the provider session list failed: {err[-300:]}")
    for line in out.splitlines():
        match = SESSION_LINE.match(line.strip())
        if match and match.group("name") == session:
            return match.group("hardware").upper()
    raise RuntimeError(f"no live session named {session}")


def execute(session: str, pinned_sha: str, steps: str, images: str, max_steps: str,
            official_limit: str, family_targets: str, timeout: float) -> tuple:
    """Run the pinned remote entry point inside the session."""
    environment = [
        f"HALA_PINNED_SOURCE_SHA={pinned_sha}",
        "HALA_FREE_COLAB_RESEARCH=1",
        "HALA_NO_PAID_RUNTIME_CONFIRMED=1",
        f"H61_STEPS={steps}",
        f"H61_IMAGES={images}",
        f"H61_MAX_STEPS={max_steps}",
        f"H61_OFFICIAL_LIMIT={official_limit}",
        f"H61_FAMILY_TARGETS={family_targets}",
        f"H62_WORK={REMOTE_WORK}",
    ]
    command = ["colab", "exec", "-s", session, "-f", str(LAUNCHER),
               "--timeout", str(int(timeout))]
    for item in environment:
        command.extend(["--env", item])
    return run(command, timeout=timeout + 600)


def download(session: str, remote_path: str, local_path: Path) -> bool:
    """Read one artefact back off the remote filesystem."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    code, out, err = run(["colab", "download", "-s", session, remote_path, str(local_path)])
    if code != 0:
        print(f"H62_DOWNLOAD_FAILED {remote_path}: {(out + err)[-300:]}", flush=True)
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="hala-h62")
    parser.add_argument("--pinned-sha", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--steps", default="export,firstparty,curriculum,baseline,train,evaluate")
    parser.add_argument("--images", default="slider=3500,slide_resize=2500,sheet=2500,text_highlight=2500")
    parser.add_argument("--max-steps", default="600")
    parser.add_argument("--official-limit", default="30")
    parser.add_argument("--family-targets", default='{"h61_firstparty_bilingual_drag_train": 2000}')
    parser.add_argument("--timeout", type=float, default=21600.0)
    args = parser.parse_args(argv)

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {
        "schema": "hala.h62.run-h61-diagnostic.v1",
        "session": args.session,
        "pinned_sha": args.pinned_sha,
        "steps": args.steps,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    try:
        report["hardware"] = session_hardware(args.session)
        if report["hardware"] != "GPU":
            raise RuntimeError(f"the session hardware is {report['hardware']}, not GPU")
        code, out, err = execute(args.session, args.pinned_sha, args.steps, args.images,
                                 args.max_steps, args.official_limit,
                                 args.family_targets, args.timeout)
        log_path = local_dir / "remote-exec.log"
        log_path.write_text(out + "\n=== STDERR ===\n" + err, encoding="utf8")
        report["exec_exit"] = code
        report["log"] = str(log_path)
        report["stdout_tail"] = out[-2000:]
        report["stderr_tail"] = err[-2000:]
        if code != 0:
            raise RuntimeError(f"the remote execution failed with exit {code}")
        for name, remote in (
            ("launch-receipt", f"{REMOTE_WORK}/h62-remote-launch-receipt.json"),
            ("run-receipt", f"{REMOTE_WORK}/run/run-receipt.json"),
        ):
            report[f"{name}_downloaded"] = download(
                args.session, remote, local_dir / f"{name}.json")
    except BaseException as error:  # noqa: BLE001
        report["failure"] = f"{type(error).__name__}: {error}"[:400]
        report["elapsed_seconds"] = round(time.time() - started, 3)
        (local_dir / "run-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf8")
        print(f"H62_DIAGNOSTIC_FAILED {type(error).__name__}: {error}", flush=True)
        return 1
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["status"] = "completed"
    (local_dir / "run-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print("H62_DIAGNOSTIC_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

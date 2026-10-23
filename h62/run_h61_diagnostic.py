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
    # Callers consume the (exit_code, stdout, stderr) triple. Returning the
    # CompletedProcess object would raise TypeError before GPU admission,
    # evaluation or receipt retrieval could complete.
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return completed.returncode, completed.stdout, completed.stderr


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
            official_limit: str, family_targets: str, timeout: float,
            stage_sha256: str = "") -> tuple:
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
    if stage_sha256:
        environment.append(f"H67_STAGE_SHA256={stage_sha256}")
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
    return local_path.is_file() and local_path.stat().st_size > 0


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
    parser.add_argument("--stage-sha256", default="")
    args = parser.parse_args(argv)

    if args.stage_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", args.stage_sha256):
            raise ValueError("invalid H67 CPU stage SHA256")
        if any(s.strip() in {"export", "firstparty", "curriculum", "baseline"}
               for s in args.steps.split(",")):
            raise ValueError("CPU preparation forbidden in staged GPU run")
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
        # The official Colab CLI renders the accelerator model (e.g. T4),
        # not necessarily the generic word GPU. Admit only the named free
        # T4 tier, never a CPU or an unverified L4/A100 paid variant.
        if report["hardware"] != "T4":
            raise RuntimeError(f"the session hardware is {report['hardware']}, not the approved T4")
        code, out, err = execute(args.session, args.pinned_sha, args.steps, args.images,
                                 args.max_steps, args.official_limit,
                                 args.family_targets, args.timeout,
                                 args.stage_sha256)
        log_path = local_dir / "remote-exec.log"
        log_path.write_text(out + "\n=== STDERR ===\n" + err, encoding="utf8")
        report["exec_exit"] = code
        report["log"] = str(log_path)
        report["stdout_tail"] = out[-2000:]
        report["stderr_tail"] = err[-2000:]
        if code != 0:
            raise RuntimeError(f"the remote execution failed with exit {code}")
        receipts = {}
        for name, remote in (
            ("launch-receipt", f"{REMOTE_WORK}/h62-remote-launch-receipt.json"),
            ("run-receipt", f"{REMOTE_WORK}/run/run-receipt.json"),
        ):
            target = local_dir / f"{name}.json"
            # A previous run's receipt is not evidence for a new launch.
            target.unlink(missing_ok=True)
            received = download(args.session, remote, target)
            report[f"{name}_downloaded"] = received
            if not received:
                raise RuntimeError(f"missing or empty remote {name}")
            try:
                receipts[name] = json.loads(target.read_text(encoding="utf8"))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"invalid remote {name}") from error

        launch_receipt = receipts["launch-receipt"]
        run_receipt = receipts["run-receipt"]
        if (not isinstance(launch_receipt, dict)
                or launch_receipt.get("status") != "completed"
                or launch_receipt.get("colab_run_exit") != 0
                or launch_receipt.get("pinned_source_sha") != args.pinned_sha
                or launch_receipt.get("repo_head") != args.pinned_sha
                or launch_receipt.get("repo_source_sha_verified") is not True):
            raise RuntimeError("remote launch receipt does not prove successful pinned execution")
        if (not isinstance(run_receipt, dict)
                or run_receipt.get("status") != "completed"
                or run_receipt.get("failures")
                or run_receipt.get("steps") != [s.strip() for s in args.steps.split(",") if s.strip()]):
            raise RuntimeError("H61 run receipt is missing success or does not match requested steps")
        report["remote_receipts_verified"] = True
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

"""H72 host-side resumable 16-MiB upload to an already-approved Colab session.

Do not claim a training result from transport alone. Exact SHA256 of each
uploaded chunk and the complete remote ZIP is independently verified before
H67 can re-derive the H61 TRAIN media and labels. No Kaggle credentials are
sent to Colab. CPU-only preparation; no provider sessions created here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from h72.remote_stage import MAX_CHUNK, ROOT, SCHEMA, read_manifest
from h72.remote_stage import sha as file_sha


def plan(source: Path, zip_sha256: str, git_sha: str):
    source = Path(source)
    if (not re.fullmatch("[0-9a-f]{64}", zip_sha256 or "")
            or not re.fullmatch("[0-9a-f]{40}", git_sha or "")):
        raise ValueError("operator must pin the original ZIP SHA and Git commit")
    if not source.is_file() or source.is_symlink():
        raise ValueError("original source-bound ZIP is missing or linked")
    parts = []
    overall = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(MAX_CHUNK)
            if not chunk:
                break
            parts.append({"name": f"chunk_{len(parts):05d}.bin",
                          "sha256": hashlib.sha256(chunk).hexdigest(),
                          "bytes": len(chunk)})
            overall.update(chunk)
    if overall.hexdigest() != zip_sha256:
        raise ValueError("reviewed stage SHA256 differs from local source")
    data = {"schema": SCHEMA, "zip_sha256": zip_sha256, "git_sha": git_sha,
            "zip_bytes": source.stat().st_size, "parts": parts}
    # Check source against exactly the same constraints applied on the VM.
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "manifest.json").write_text(json.dumps(data), encoding="utf8")
        read_manifest(Path(td))
    return data


def executable():
    found = shutil.which("colab")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "colab"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("official colab CLI is not installed on host")


def _run(cli, *args, timeout=600, runner=subprocess.run):
    command = [cli, *args]
    r = runner(command, capture_output=True, text=True, errors="replace", timeout=timeout)
    if r.returncode:
        # Never print URL/query strings: provider failures may include
        # short-lived proxy bearer tokens in their diagnostics.
        raise RuntimeError(f"H72 Colab command failed: {args[0]} (exit {r.returncode})")
    return r.stdout


def _marker(output: str, key: str):
    for line in output.splitlines():
        if line.startswith(key + "="):
            return json.loads(line.split("=", 1)[1])
    raise ValueError("remote Colab response omitted cryptographic proof marker")


def transfer(source: Path, zip_sha: str, git_sha: str, session: str, *,
             cpu_test=False, runner=subprocess.run):
    if not re.fullmatch("[a-zA-Z0-9_-]{1,80}", session):
        raise ValueError("trusted existing session name required")
    data = plan(source, zip_sha, git_sha)
    from h62.run_h61_diagnostic import SESSION_LINE
    cli = executable()
    sessions = _run(cli, "sessions", runner=runner)
    hardware = None
    for line in sessions.splitlines():
        match = SESSION_LINE.match(line.strip())
        if match and match.group("name") == session:
            hardware = match.group("hardware").upper()
            break
    if hardware != "T4" and not (cpu_test and hardware == "CPU"):
        raise RuntimeError("must use an existing authorized Colab T4")
    script = Path(__file__).with_name("remote_stage.py")
    if not script.is_file():
        raise FileNotFoundError("H72 independently reviewed VM verifier unavailable")
    with tempfile.TemporaryDirectory(prefix=".hala-h72-host-") as scratch:
        folder = Path(scratch)
        receipt = folder / "manifest.json"
        receipt.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), "utf8")
        prepared = _run(cli, "exec", "-s", session, "-f", str(script),
                        "--env", "H72_ACTION=prepare", "--timeout", "90", runner=runner)
        if "H72_ROOT_PREPARED" not in prepared:
            raise ValueError("Colab did not create H72 staging directory")
        _run(cli, "upload", "-s", session, str(receipt), str(ROOT / "manifest.json"),
             runner=runner)
        inventory = _run(cli, "exec", "-s", session, "-f", str(script),
                         "--env", "H72_ACTION=probe", "--timeout", "180", runner=runner)
        present = _marker(inventory, "H72_HAVE_JSON")
        if not isinstance(present, list):
            raise ValueError("untrusted remote inventory")
        good = set(present)
        valid = {p["name"] for p in data["parts"]}
        if not good <= valid:
            raise ValueError("untrusted remote uploaded part list")
        with Path(source).open("rb") as original:
            for part in data["parts"]:
                block = original.read(part["bytes"])
                if part["name"] in good:
                    continue
                if len(block) != part["bytes"] or hashlib.sha256(block).hexdigest() != part["sha256"]:
                    raise ValueError("local ZIP content changed during transport")
                chunk_file = folder / part["name"]
                chunk_file.write_bytes(block)
                try:
                    _run(cli, "upload", "-s", session, str(chunk_file),
                         str(ROOT / part["name"]), runner=runner)
                finally:
                    chunk_file.unlink(missing_ok=True)
            # Any skipped remote part must not cause a misaligned host stream.
        result = _run(cli, "exec", "-s", session, "-f", str(script),
                      "--env", "H72_ACTION=assemble", "--timeout", "600",
                      timeout=900, runner=runner)
        final = _marker(result, "H72_STAGE_READY")
        if (not isinstance(final, dict) or final.get("status") != "H72_STAGE_READY"
                or final.get("sha256") != zip_sha
                or final.get("git_sha") != git_sha
                or final.get("bytes") != data["zip_bytes"]):
            raise ValueError("remote ZIP did not pass pinned archive SHA verification")
        return {"schema": SCHEMA, "status": "REMOTE_ZIP_SHA_VERIFIED",
                "zip_sha256": zip_sha, "git_sha": git_sha,
                "bytes": data["zip_bytes"], "chunks": len(data["parts"]),
                "resumed_verified_chunks": len(good),
                "gpu_training_executed": False,
                "h61_source_rederivation_executed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--git-sha", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--cpu-smoke", action="store_true")
    args = p.parse_args()
    print(json.dumps(transfer(args.source, args.sha256, args.git_sha,
                              args.session, cpu_test=args.cpu_smoke), sort_keys=True))


if __name__ == "__main__":
    main()

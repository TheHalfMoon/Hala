"""H65: pull a SHA-verified immutable training checkpoint before Colab expires.

Runs on the *host* (CPU), never allocates GPUs, never executes weights and
never modifies the running Colab session. Colab CLI only downloads single
files, so read the published pointer, download exactly its signed artifact
inventory, and admit it using the existing training checkpoint validator.

The locally retained bundle is a recovery staging copy, NOT provider durable
storage. The caller must separately publish it to approved private storage
with authenticated readback before deleting the staging copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from training.checkpoints import atomic_json, load_checkpoint, validate_version
from training.artifacts import strict_json

SCHEMA = "hala.h65.host-checkpoint-egress.v1"
REMOTE_TRAINING_ROOT = "/content/hala-h62-work/run/training"
VERSION = re.compile(r"^checkpoints/step-([0-9]{8})-[0-9a-f]{32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MEMBER = re.compile(r"^[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$")
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download(session, remote_path, target, *, runner):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    command = ["colab", "download", "-s", session, remote_path, str(target)]
    try:
        result = runner(command, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        return False
    if target.stat().st_size < 1 or target.stat().st_size > MAX_ARTIFACT_BYTES:
        target.unlink(missing_ok=True)
        return False
    return True


def _strict_read(path, limit):
    if path.stat().st_size > limit:
        raise ValueError("checkpoint metadata exceeds size bound")
    return strict_json(path.read_text("utf8"))


def _pointer(path):
    pointer = _strict_read(path, 1024)
    if (not isinstance(pointer, dict)
            or set(pointer) != {"schema", "version", "checkpoint_sha256"}
            or pointer["schema"] != "hala.training.latest.v1"
            or not isinstance(pointer["version"], str)
            or VERSION.fullmatch(pointer["version"]) is None
            or not isinstance(pointer["checkpoint_sha256"], str)
            or HEX64.fullmatch(pointer["checkpoint_sha256"]) is None):
        raise ValueError("remote latest pointer is invalid or unsafe")
    return pointer


def _member_name(value):
    if (not isinstance(value, str) or len(value) > 240
            or MEMBER.fullmatch(value) is None
            or any(segment in (".", "..") for segment in value.split("/"))):
        raise ValueError("checkpoint manifest contains an unsafe artifact name")
    return value


def snapshot_once(session: str, destination: Path, *, runner=subprocess.run,
                  remote_root: str = REMOTE_TRAINING_ROOT) -> dict:
    """Attempt one read-only recovery of the newest fully published checkpoint."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", session):
        raise ValueError("unsafe Colab session identifier")
    if remote_root != REMOTE_TRAINING_ROOT:
        raise ValueError("only the H61 training output may be salvaged")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".h65-egress-") as temp:
        temp = Path(temp)
        temporary_pointer = temp / "latest.json"
        if not _download(session, remote_root + "/latest.json", temporary_pointer,
                         runner=runner):
            return {"schema": SCHEMA, "status": "NO_PUBLISHED_CHECKPOINT"}
        pointer = _pointer(temporary_pointer)
        relative = pointer["version"]
        step = int(VERSION.fullmatch(relative).group(1))
        existing_pointer = destination / "latest.json"
        if existing_pointer.exists():
            prior_version, prior_state = load_checkpoint(destination, None)
            prior = _pointer(existing_pointer)
            if prior_state["step"] > step:
                return {"schema": SCHEMA, "status": "OLDER_REMOTE_SNAPSHOT",
                        "retained_step": prior_state["step"]}
            if prior_state["step"] == step:
                if prior["checkpoint_sha256"] != pointer["checkpoint_sha256"]:
                    raise ValueError("forked checkpoint at an identical training step")
                return {"schema": SCHEMA, "status": "ALREADY_VERIFIED", "step": step}

        metadata_path = temp / "checkpoint.json"
        if not _download(session, remote_root + "/" + relative + "/checkpoint.json",
                         metadata_path, runner=runner):
            return {"schema": SCHEMA, "status": "METADATA_UNAVAILABLE"}
        if metadata_path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("checkpoint metadata is oversized")
        if digest(metadata_path) != pointer["checkpoint_sha256"]:
            raise ValueError("checkpoint metadata digest mismatches atomic remote pointer")
        metadata = _strict_read(metadata_path, MAX_METADATA_BYTES)
        if (not isinstance(metadata, dict)
                or set(metadata) != {"schema", "state", "artifacts"}
                or metadata["schema"] != "hala.training.checkpoint.v1"
                or not isinstance(metadata["artifacts"], dict)
                or not 1 <= len(metadata["artifacts"]) <= 1024):
            raise ValueError("invalid checkpoint manifest")
        if metadata["state"].get("step") != step:
            raise ValueError("checkpoint pointer step differs from manifest")
        artifacts = metadata["artifacts"]
        for name, expected in artifacts.items():
            _member_name(name)
            if not isinstance(expected, str) or HEX64.fullmatch(expected) is None:
                raise ValueError("checkpoint artifact has invalid digest")
        if "resume.pt" not in artifacts or not any(
                item.startswith("adapter/") and item.endswith(".safetensors")
                for item in artifacts):
            raise ValueError("checkpoint cannot restore complete optimizer and adapter")

        staging_parent = destination / "checkpoints"
        staging_parent.mkdir(parents=True, exist_ok=True)
        stage = staging_parent / (".staging-h65-" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            shutil.copyfile(metadata_path, stage / "checkpoint.json")
            total_bytes = metadata_path.stat().st_size
            for name, expected in sorted(artifacts.items()):
                local = stage / name
                remote = remote_root + "/" + relative + "/" + name
                if not _download(session, remote, local, runner=runner):
                    raise IOError("checkpoint artifact download incomplete")
                length = local.stat().st_size
                total_bytes += length
                if length > MAX_ARTIFACT_BYTES or total_bytes > MAX_TOTAL_BYTES:
                    raise ValueError("checkpoint egress exceeds bounded byte budget")
                if digest(local) != expected:
                    raise ValueError("checkpoint artifact SHA-256 mismatch")
            state, _ = validate_version(stage, None)
            if state["step"] != step:
                raise ValueError("inconsistent validated checkpoint step")
            target = destination / relative
            if target.exists():
                raise ValueError("target version already exists unexpectedly")
            os.replace(stage, target)
            atomic_json(destination / "latest.json", pointer)
            # Re-read everything through the original independent validator.
            version, confirmed = load_checkpoint(destination, None)
            return {"schema": SCHEMA, "status": "VERIFIED_HOST_STAGING",
                    "step": confirmed["step"], "checkpoint_sha256": pointer["checkpoint_sha256"],
                    "artifacts": len(artifacts), "downloaded_bytes": total_bytes,
                    "storage_kind": "host_local_only_not_provider_readback",
                    "version": version.relative_to(destination).as_posix()}
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    result = snapshot_once(args.session, args.destination)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("VERIFIED_HOST_STAGING", "ALREADY_VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())

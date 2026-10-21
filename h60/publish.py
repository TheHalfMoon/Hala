"""H60 private provider publication and readback verification.

A shard is only "stored" after the provider itself reports the shard, the
dataset is confirmed private for the owning account, an unauthenticated
request is rejected, and every published byte is read back and matched against
the locally computed SHA-256 manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

OWNER = "abdulazizshehri"
KAGGLE_API = "https://www.kaggle.com/api/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"kaggle {' '.join(args)} failed: {result.stdout[-500:]} {result.stderr[-500:]}"
        )
    return result


def api_get(path: str, token: str) -> dict:
    request = urllib.request.Request(
        KAGGLE_API + path, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def anonymous_status(url: str) -> int:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def dataset_files(ref: str) -> list[tuple[str, int]]:
    """Provider file listing as (relative name, size) pairs."""
    result = _kaggle("datasets", "files", ref)
    rows: list[tuple[str, int]] = []
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            size = int(parts[1])
        except ValueError:
            continue
        rows.append((name, size))
    return rows


def wait_for_dataset(ref: str, timeout: float = 3600.0) -> list[tuple[str, int]]:
    deadline = time.time() + timeout
    last: list[tuple[str, int]] = []
    while time.time() < deadline:
        try:
            last = dataset_files(ref)
            if last:
                return last
        except RuntimeError:
            pass
        time.sleep(15)
    raise TimeoutError(f"provider never published a file listing for {ref}: {last}")


def create_private_dataset(upload_dir: Path, ref: str, title: str, subtitle: str) -> None:
    """Create a NEW private dataset from an upload directory."""
    if not ref.startswith(OWNER + "/"):
        raise ValueError("only the owning account namespace may be used")
    metadata = upload_dir / "dataset-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "title": title,
                "id": ref,
                "licenses": [{"name": "Apache-2.0"}],
                "isPrivate": True,
                "subtitle": subtitle,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    _kaggle("datasets", "create", "-p", str(upload_dir), "-r", "skip")


def download_dataset(ref: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    _kaggle("datasets", "download", "-p", str(destination), ref)
    archives = [item for item in destination.iterdir() if item.is_file()]
    if not archives:
        raise RuntimeError(f"provider download produced no archive for {ref}")
    return max(archives, key=lambda item: item.stat().st_size)


def verify_private(ref: str, token: str) -> dict:
    """Authenticated privacy flag plus an unauthenticated rejection check."""
    view = api_get(f"/datasets/view/{ref}", token)
    is_private = bool(view.get("isPrivate"))
    anonymous = anonymous_status(f"{KAGGLE_API}/datasets/view/{ref}")
    return {
        "provider_is_private": is_private,
        "provider_view_total_bytes": view.get("totalBytes"),
        "provider_version": view.get("currentVersionNumber"),
        "anonymous_view_http_status": anonymous,
    }


def verify_readback(
    ref: str,
    token: str,
    expected_files: dict[str, int],
    expected_shas: dict[str, str],
    work_dir: Path,
) -> dict:
    """Prove the provider holds exactly the published bytes.

    1. The authenticated provider listing must match the local file inventory
       by name and exact byte size.
    2. The provider download must be extracted in memory and every member
       must hash to the locally recorded SHA-256.
    """
    listing = dict(wait_for_dataset(ref))
    if listing != expected_files:
        missing = sorted(set(expected_files) - set(listing))
        extra = sorted(set(listing) - set(expected_files))
        size_mismatch = sorted(
            name
            for name in set(expected_files) & set(listing)
            if expected_files[name] != listing[name]
        )
        raise ValueError(
            "provider file listing differs from the published inventory: "
            f"missing={missing[:5]} extra={extra[:5]} size_mismatch={size_mismatch[:5]}"
        )
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dataset(ref, work_dir)
    archive_sha = sha256_file(archive)
    checked = 0
    mismatched: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if names != set(expected_shas):
            raise ValueError("provider download archive does not hold the published members")
        for name in sorted(names):
            with bundle.open(name) as member:
                digest = hashlib.sha256()
                for chunk in iter(lambda: member.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_shas[name]:
                mismatched.append(name)
            checked += 1
    if mismatched:
        raise ValueError(f"provider readback bytes differ for {len(mismatched)} members")
    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "provider_readback_archive_sha256": archive_sha,
        "provider_readback_members_checked": checked,
        "provider_readback_bytes_mismatched": 0,
        "provider_listing_files": len(listing),
    }


def delete_dataset(ref: str) -> None:
    _kaggle("datasets", "delete", "-p", "yes", ref, check=False)

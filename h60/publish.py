"""H60 private provider publication and byte-exact readback verification.

A shard is only "stored" after the provider itself reports the shard, the
dataset is confirmed private for the owning account, an unauthenticated
request is rejected with HTTP 403, and every published byte is read back from
the provider and matched against the locally computed SHA-256 manifest.

The provider's own REST endpoints are used for the readback because the
provider's CLI host is aggressively throttled for large datasets: the dataset
download is streamed once and every member is hashed locally, which is a
stronger proof than an index listing and costs one request instead of one
request per 200 files.
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
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
BLOCK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the provider CLI with backoff for throttled or transient failures.

    The provider CLI writes progress output that is not always valid in the
    console code page, so the captured bytes are decoded as UTF-8 with
    replacement instead of the Windows ANSI code page. A decoding error must
    never be mistaken for a failed provider call.
    """
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        result = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return result
        text = f"{result.stdout[-400:]} {result.stderr[-400:]}"
        last = RuntimeError(f"kaggle {' '.join(args)} failed: {text}")
        if "429" not in text and "Too Many Requests" not in text and attempt >= 2:
            break
        time.sleep(min(60.0, 5.0 * (attempt + 1)))
    if check:
        raise last  # type: ignore[misc]
    return result


def _rest_get(path: str, token: str, accept: str = "application/json", timeout: int = 120):
    request = urllib.request.Request(
        KAGGLE_API + path,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
            "Accept": accept,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def api_get(path: str, token: str) -> dict:
    return json.loads(_rest_get(path, token).decode())


def anonymous_status(url: str) -> int:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def dataset_files(ref: str, token: str, max_pages: int = 400) -> tuple[dict[str, int], bool]:
    """Provider file index, paginated.

    Returns the index and whether the enumeration reached the end of the
    provider index. The index is a secondary receipt only: the authoritative
    proof that the provider holds the published bytes is the byte-level
    readback in `verify_readback`.
    """
    files: dict[str, int] = {}
    token_page: str | None = None
    for _ in range(max_pages):
        suffix = f"/datasets/list/{ref}?pageSize=200"
        if token_page:
            suffix += f"&pageToken={token_page}"
        try:
            payload = json.loads(_rest_get(suffix, token).decode())
        except urllib.error.HTTPError as error:
            if error.code in RETRYABLE_STATUS:
                time.sleep(10.0)
                continue
            raise
        for row in payload.get("datasetFiles") or []:
            name = row.get("name")
            if name:
                files[name] = int(row.get("totalBytes") or 0)
        token_page = payload.get("nextPageTokenNullable")
        if not token_page:
            return files, True
    return files, False


def wait_for_dataset(ref: str, token: str, timeout: float = 3600.0) -> dict[str, int]:
    deadline = time.time() + timeout
    last: dict[str, int] = {}
    while time.time() < deadline:
        try:
            last, complete = dataset_files(ref, token, max_pages=1)
            if last:
                return last
        except Exception:  # noqa: BLE001
            pass
        time.sleep(15)
    raise TimeoutError(f"provider never published a file listing for {ref}: {len(last)}")


def wait_until_visible(ref: str, token: str, timeout: float = 1800.0) -> dict:
    """Wait until the provider serves the freshly created dataset.

    A dataset that was just created is not immediately readable through the
    provider API: an immediate read is refused with HTTP 403 while the
    provider still processes the new version, and it is briefly absent from
    the provider index as HTTP 404. This gate waits for the provider itself
    to serve the dataset before any privacy or readback proof is attempted, so
    a propagation delay is never recorded as a privacy failure.
    """
    deadline = time.time() + timeout
    last = "not attempted"
    while time.time() < deadline:
        try:
            return api_get(f"/datasets/view/{ref}", token)
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}"
            if error.code not in RETRYABLE_STATUS and error.code not in (403, 404):
                raise
        except Exception as error:  # noqa: BLE001
            last = repr(error)
        time.sleep(10.0)
    raise TimeoutError(f"provider never served the new dataset {ref}: {last}")


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


def download_dataset(ref: str, destination: Path, token: str) -> Path:
    """Stream the provider dataset archive to disk with backoff."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "provider-dataset.zip"
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(
                f"{KAGGLE_API}/datasets/download/{ref}",
                headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=1800) as response:
                with target.open("wb") as handle:
                    while True:
                        chunk = response.read(BLOCK)
                        if not chunk:
                            break
                        handle.write(chunk)
            if target.stat().st_size == 0:
                raise IOError("provider returned an empty dataset archive")
            return target
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in RETRYABLE_STATUS:
                raise
        except Exception as error:  # noqa: BLE001
            last = error
        time.sleep(min(90.0, 10.0 * (attempt + 1)))
    raise RuntimeError(f"provider download failed for {ref}: {last}")


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


def _common_prefix(names: list[str]) -> str:
    """Longest common directory prefix of the provider archive members."""
    if not names:
        return ""
    prefix = os.path.commonprefix(names)
    if "/" not in prefix:
        return ""
    return prefix[: prefix.rfind("/") + 1]


def verify_readback(
    ref: str,
    token: str,
    expected_files: dict[str, int],
    expected_shas: dict[str, str],
    work_dir: Path,
) -> dict:
    """Prove the provider holds exactly the published bytes.

    1. The provider archive is downloaded from the provider.
    2. Every member is enumerated and hashed locally.
    3. The member set, every member size and every member SHA-256 must match
       the locally computed inventory exactly.
    """
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = download_dataset(ref, work_dir, token)
    archive_sha = sha256_file(archive)
    checked = 0
    mismatched: list[str] = []
    size_mismatched: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        archive_names = set(bundle.namelist())
        # Map every provider archive member onto the published member name.
        # The provider may wrap an expanded tree in one common directory.
        mapping = {name: name for name in archive_names}
        if archive_names != set(expected_shas):
            prefix = _common_prefix(sorted(archive_names))
            stripped = {
                name: name[len(prefix) :]
                for name in archive_names
                if prefix and name.startswith(prefix)
            }
            if set(stripped.values()) != set(expected_shas):
                missing = sorted(set(expected_shas) - set(stripped.values()))
                extra = sorted(set(stripped.values()) - set(expected_shas))
                raise ValueError(
                    "provider archive does not hold exactly the published members: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            mapping = stripped
        for archive_name, member_name in sorted(mapping.items()):
            info = bundle.getinfo(archive_name)
            if info.file_size != expected_files[member_name]:
                size_mismatched.append(member_name)
                continue
            with bundle.open(archive_name) as member:
                digest = hashlib.sha256()
                for chunk in iter(lambda: member.read(BLOCK), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_shas[member_name]:
                mismatched.append(member_name)
            checked += 1
    if size_mismatched:
        raise ValueError(
            f"provider readback differs in size for {len(size_mismatched)} members"
        )
    if mismatched:
        raise ValueError(f"provider readback bytes differ for {len(mismatched)} members")
    shutil.rmtree(work_dir, ignore_errors=True)
    return {
        "provider_readback_archive_sha256": archive_sha,
        "provider_readback_members_checked": checked,
        "provider_readback_bytes_mismatched": 0,
        "provider_readback_files": len(expected_shas),
    }


def delete_dataset(ref: str) -> None:
    """Remove one of this project's own shard datasets.

    Only the H60 shard namespace may be deleted: the founder's earlier
    private datasets are never touched by this pipeline.
    """
    if not ref.startswith(f"{OWNER}/hala-h60-dragon-train-"):
        raise ValueError("only the H60 shard namespace may be deleted")
    _kaggle("datasets", "delete", "-y", ref, check=False)


def dataset_exists(ref: str, token: str) -> bool:
    try:
        api_get(f"/datasets/view/{ref}", token)
        return True
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return False
        raise

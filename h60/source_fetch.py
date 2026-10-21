"""H60 bounded, resumable retrieval of pinned original DragOn TRAIN byte windows.

Every window is fetched with explicit HTTP Range requests against the pinned
immutable source revision. A window is only accepted when the provider returns
an actual 206 response with the exact expected Content-Range and the exact
expected byte count. Per-window SHA-256 receipts are written incrementally so
an interrupted run never has to restart from byte zero.

A pinned window is 512 MiB, so each window is fetched as ordered parallel
sub-ranges of `SUBRANGE_BYTES`. Memory stays bounded by the number of
in-flight sub-ranges instead of by the window size, and the window SHA-256 is
still computed over the exact pinned byte range in archive order.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

REPO = "Hcompany/DragOn"
REV = "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
USER_AGENT = "Hala-H60-bounded-immutable-TRAIN-source-2026-10-21"
BLOCK = 1 << 20
SUBRANGE_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS = 4
# Test seam: a local Range-capable origin may serve a synthetic archive.
DEFAULT_SOURCE_BASE_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/train"


def source_url(domain: str) -> str:
    base = os.environ.get("HALA_H60_SOURCE_BASE_URL", DEFAULT_SOURCE_BASE_URL)
    return f"{base.format(repo=REPO, rev=REV)}/{domain}.tar"


def fetch_range(url: str, start: int, end: int, total: int) -> bytes:
    """Fetch one exact inclusive byte range and prove the provider honoured it."""
    expected_range = f"bytes {start}-{end}/{total}"
    expected_length = end - start + 1
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if response.status != 206:
                    raise ValueError("source provider did not answer with HTTP 206")
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("source provider returned an unexpected Content-Range")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != expected_length:
                    raise ValueError("source provider returned an unexpected Content-Length")
                buffer = bytearray()
                while len(buffer) < expected_length:
                    chunk = response.read(min(BLOCK, expected_length - len(buffer)))
                    if not chunk:
                        break
                    buffer += chunk
                if len(buffer) != expected_length:
                    raise IOError("source provider truncated the requested range")
                return bytes(buffer)
        except (urllib.error.URLError, IOError, ValueError) as error:  # noqa: PERF203
            last_error = error
            time.sleep(min(30.0, 2.0 ** attempt))
    raise IOError(f"bounded source range {start}-{end} failed: {last_error}")


class WindowedSource:
    """Ordered, prefetching reader over the pinned window inventory.

    `windows` is the ordered list of window descriptors for one domain. The
    reader yields the exact bytes of each window in order, computes the
    per-window SHA-256 and a running whole-archive SHA-256, and records one
    immutable receipt per window.
    """

    def __init__(self, domain: str, windows: list[dict], total_bytes: int, receipt_dir: Path):
        self.domain = domain
        self.windows = windows
        self.total_bytes = total_bytes
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.url = source_url(domain)
        self._archive_digest = hashlib.sha256()
        self._archive_bytes = 0
        self.window_receipts: list[dict] = []

    def _receipt_path(self, window: dict) -> Path:
        return self.receipt_dir / f"{window['id']}.receipt.json"

    def _load_receipt(self, window: dict) -> dict | None:
        path = self._receipt_path(window)
        if not path.exists():
            return None
        try:
            receipt = json.loads(path.read_text("utf8"))
        except (OSError, ValueError):
            return None
        if (
            receipt.get("schema") != "hala.h60.train-tar-window-receipt.v1"
            or receipt.get("original_window_id") != window["id"]
            or receipt.get("source_domain") != self.domain
            or receipt.get("source_commit_sha") != REV
            or receipt.get("range_start") != window["byte_start"]
            or receipt.get("range_end_inclusive") != window["byte_end_inclusive"]
            or receipt.get("bytes_verified") != window["expected_length"]
        ):
            return None
        return receipt

    def _write_receipt(self, window: dict, digest: str) -> dict:
        receipt = {
            "schema": "hala.h60.train-tar-window-receipt.v1",
            "original_window_id": window["id"],
            "source_repo": REPO,
            "source_commit_sha": REV,
            "source_domain": self.domain,
            "split": "train",
            "range_start": window["byte_start"],
            "range_end_inclusive": window["byte_end_inclusive"],
            "expected_content_range": window["expected_content_range"],
            "bytes_verified": window["expected_length"],
            "original_window_sha256": digest,
            "http_status": 206,
            "original_tar_lfs_sha256": window["upstream_full_tar_lfs_sha256"],
            "whole_archive_sha256_complete": False,
            "model_gpu_steps": 0,
        }
        self._receipt_path(window).write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
        )
        return receipt

    def stream(self, workers: int = 6) -> Iterator[tuple[dict, bytes]]:
        """Yield `(window, payload)` in archive order with bounded memory."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for window in self.windows:
                start = window["byte_start"]
                end = window["byte_end_inclusive"]
                spans = []
                cursor = start
                while cursor <= end:
                    span_end = min(cursor + SUBRANGE_BYTES - 1, end)
                    spans.append((cursor, span_end))
                    cursor = span_end + 1
                futures = [
                    pool.submit(fetch_range, self.url, span_start, span_end, self.total_bytes)
                    for span_start, span_end in spans
                ]
                window_digest = hashlib.sha256()
                for future in futures:
                    payload = future.result()
                    window_digest.update(payload)
                    self._archive_digest.update(payload)
                    self._archive_bytes += len(payload)
                    yield window, payload
                digest = window_digest.hexdigest()
                receipt = self._load_receipt(window)
                if receipt is None or receipt.get("original_window_sha256") != digest:
                    receipt = self._write_receipt(window, digest)
                self.window_receipts.append(
                    {
                        "original_window_id": window["id"],
                        "range_start": window["byte_start"],
                        "range_end_inclusive": window["byte_end_inclusive"],
                        "bytes_verified": window["expected_length"],
                        "original_window_sha256": digest,
                    }
                )

    @property
    def archive_sha256(self) -> str:
        return self._archive_digest.hexdigest()

    @property
    def archive_bytes(self) -> int:
        return self._archive_bytes

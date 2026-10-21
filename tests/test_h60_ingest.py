"""H60 synthetic end-to-end tests for the original DragOn TRAIN ingest.

These tests never touch the real upstream source and never touch a provider
account. They build a synthetic TAR archive with the exact original member
layout, serve it from a local Range-capable origin and prove that the H60
ingest pairs every original image with its original task, keeps the source
ordering, verifies the whole-archive SHA-256 and produces a real shard.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import tarfile
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

DOMAIN = "slider"
IMAGE_COUNT = 40
TASKS_PER_IMAGE = 3
WINDOW = 64 * 1024


def _jpeg(colour: tuple[int, int, int], size: tuple[int, int] = (96, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def _tar_member(name: str, payload: bytes) -> tuple[bytes, bytes]:
    """One 512-byte-aligned ustar member plus its mandatory block padding."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    header = info.tobuf(tarfile.USTAR_FORMAT)
    padding = b"\0" * ((512 - (len(payload) % 512)) % 512)
    return header, payload + padding


def build_synthetic_archive() -> tuple[bytes, dict]:
    """Synthetic original-style TRAIN archive plus its expected pairing."""
    image_blocks: list[bytes] = []
    task_blocks: list[bytes] = []
    expected: dict[str, dict] = {}
    for index in range(IMAGE_COUNT):
        key = str(uuid.UUID(int=index))
        payload = _jpeg((index * 5 % 255, 40, 200 - index))
        image_blocks.append(b"".join(_tar_member(f"{DOMAIN}/images/{key}.jpg", payload)))
        tasks = [
            {
                "image_id": key,
                "domain": DOMAIN,
                "ordered": index % 2 == 0,
                "intent": f"Original drag instruction number {index}",
                "start_bbox": [10, 10, 30, 30],
                "end_bbox": [40, 20, 70, 50],
                "subtype": "synthetic_probe",
            }
            for _ in range(TASKS_PER_IMAGE)
        ]
        blob = json.dumps(tasks).encode("utf8")
        task_blocks.append(b"".join(_tar_member(f"{DOMAIN}/tasks/{key}.json", blob)))
        expected[key] = {"image_sha256": hashlib.sha256(payload).hexdigest(), "tasks": tasks}
    archive = b"".join(image_blocks) + b"".join(task_blocks)
    archive += b"\0" * 1024
    return archive, expected


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *args):  # noqa: D102 - silence the test origin
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        total = len(self.payload)
        header = self.headers.get("Range")
        if not header:
            self.send_response(200)
            self.send_header("Content-Length", str(total))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        unit, _, spec = header.partition("=")
        if unit != "bytes" or "-" not in spec:
            self.send_error(416)
            return
        start_text, _, end_text = spec.partition("-")
        start = int(start_text)
        end = int(end_text) if end_text else total - 1
        if start < 0 or end >= total or start > end:
            self.send_error(416)
            return
        chunk = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


class H60SyntheticIngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive, cls.expected = build_synthetic_archive()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        cls.server.RequestHandlerClass.payload = cls.archive
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _plan(self) -> dict:
        total = len(self.archive)
        windows = []
        start = 0
        index = 0
        while start < total:
            length = min(WINDOW, total - start)
            windows.append(
                {
                    "id": f"h60-test-{DOMAIN}-{index:04d}",
                    "source_domain": DOMAIN,
                    "url": f"{self.base}/{DOMAIN}.tar",
                    "byte_start": start,
                    "byte_end_inclusive": start + length - 1,
                    "expected_length": length,
                    "total_tar_bytes": total,
                    "upstream_full_tar_lfs_sha256": hashlib.sha256(self.archive).hexdigest(),
                    "expected_content_range": f"bytes {start}-{start + length - 1}/{total}",
                }
            )
            start += length
            index += 1
        return {
            "schema": "hala.h59.full-million-drag-train-byte-range-plan.v1",
            "source_domains": {
                DOMAIN: {
                    "bytes": total,
                    "upstream_lfs_sha256": hashlib.sha256(self.archive).hexdigest(),
                }
            },
            "windows": windows,
        }

    def test_synthetic_archive_pairs_every_original_image_with_its_task(self) -> None:
        from h60 import ingest

        previous = os.environ.get("HALA_H60_SOURCE_BASE_URL")
        os.environ["HALA_H60_SOURCE_BASE_URL"] = self.base
        try:
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary) / "staging"
                result = ingest.ingest_domain(
                    DOMAIN,
                    self._plan(),
                    staging,
                    64 * 1024,
                    workers=3,
                    sample_structure_every=1,
                    sample_pixels_every=5,
                    do_publish=False,
                    token="",
                )
                self._assert_result(result)
                self._assert_shards(staging)
        finally:
            if previous is None:
                os.environ.pop("HALA_H60_SOURCE_BASE_URL", None)
            else:
                os.environ["HALA_H60_SOURCE_BASE_URL"] = previous

    def _assert_result(self, result: dict) -> None:
        self.assertTrue(result["source_tar_sha256_matches_official"])
        self.assertEqual(result["source_tar_sha256"], hashlib.sha256(self.archive).hexdigest())
        self.assertEqual(result["original_images"], IMAGE_COUNT)
        self.assertEqual(result["images_paired_with_original_tasks"], IMAGE_COUNT)
        self.assertEqual(result["images_without_original_tasks"], 0)
        self.assertEqual(result["orphan_task_members"], 0)
        self.assertEqual(result["native_drag_actions"], IMAGE_COUNT * TASKS_PER_IMAGE)
        self.assertEqual(result["rejected_rows"], 0)
        self.assertEqual(result["task_members_seen_in_full_stream"], IMAGE_COUNT)
        self.assertFalse(result["partial_run"])
        self.assertGreaterEqual(result["shard_count"], 1)

    def _assert_shards(self, staging: Path) -> None:
        import zipfile

        shards = sorted(staging.glob("h60-dragon-train-*.zip"))
        self.assertTrue(shards)
        seen_rows: dict[str, dict] = {}
        seen_images: set[str] = set()
        for shard in shards:
            with zipfile.ZipFile(shard) as bundle:
                names = set(bundle.namelist())
                self.assertIn("labels.jsonl", names)
                self.assertIn("manifest.json", names)
                for name in names:
                    if name.startswith("images/"):
                        seen_images.add(name)
                    if name.startswith("tasks/"):
                        key = Path(name).stem
                        self.assertTrue(key in self.expected)
                        payload = bundle.read(name)
                        original = json.loads(payload.decode("utf8"))
                        self.assertEqual(original, self.expected[key]["tasks"])
                for line in bundle.read("labels.jsonl").decode("utf8").splitlines():
                    row = json.loads(line)
                    seen_rows[row["id"]] = row
        self.assertEqual(len(seen_images), IMAGE_COUNT)
        self.assertEqual(len(seen_rows), IMAGE_COUNT * TASKS_PER_IMAGE)
        for row in seen_rows.values():
            key = row["image_id"]
            self.assertEqual(row["image_sha256"], self.expected[key]["image_sha256"])
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["action_type"], "drag")
            self.assertEqual(row["completion"]["coordinate_space"], "source_image_pixels")
            self.assertIn(row["ordered"], (True, False))

    def test_window_receipts_are_written_for_every_planned_window(self) -> None:
        from h60 import ingest

        previous = os.environ.get("HALA_H60_SOURCE_BASE_URL")
        os.environ["HALA_H60_SOURCE_BASE_URL"] = self.base
        try:
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary) / "staging"
                result = ingest.ingest_domain(
                    DOMAIN,
                    self._plan(),
                    staging,
                    64 * 1024,
                    workers=2,
                    sample_structure_every=0,
                    sample_pixels_every=0,
                    do_publish=False,
                    token="",
                )
                receipts = sorted((staging / "window-receipts" / DOMAIN).glob("*.receipt.json"))
                self.assertEqual(len(receipts), len(self._plan()["windows"]))
                for path in receipts:
                    receipt = json.loads(path.read_text("utf8"))
                    self.assertEqual(receipt["schema"], "hala.h60.train-tar-window-receipt.v1")
                    self.assertEqual(receipt["source_domain"], DOMAIN)
                    self.assertEqual(receipt["http_status"], 206)
                    self.assertEqual(receipt["model_gpu_steps"], 0)
                    self.assertEqual(len(receipt["original_window_sha256"]), 64)
        finally:
            if previous is None:
                os.environ.pop("HALA_H60_SOURCE_BASE_URL", None)
            else:
                os.environ["HALA_H60_SOURCE_BASE_URL"] = previous
        self.assertFalse(result["source_tar_sha256_matches_official"] is False)


class H60TarStreamTests(unittest.TestCase):
    def test_members_that_straddle_chunk_boundaries_are_never_split(self) -> None:
        from h60.tar_stream import TarStreamReader

        payloads = {
            f"{DOMAIN}/images/{uuid.UUID(int=1)}.jpg": _jpeg((1, 2, 3)),
            f"{DOMAIN}/tasks/{uuid.UUID(int=1)}.json": b'[{"image_id":"x"}]',
        }
        blob = b"".join(
            b"".join(_tar_member(name, payload)) for name, payload in payloads.items()
        ) + b"\0" * 1024
        seen: dict[str, bytes] = {}

        def keep(member) -> bool:
            return member.name in payloads

        def sink(member, payload) -> None:
            seen[member.name] = payload

        for chunk_size in (1, 7, 512, 513, 4096):
            seen.clear()
            reader = TarStreamReader(keep, sink)
            for index in range(0, len(blob), chunk_size):
                reader.feed(blob[index : index + chunk_size])
            reader.close()
            self.assertEqual(seen, payloads, f"chunk size {chunk_size}")

    def test_bytes_that_only_look_like_a_header_are_ignored(self) -> None:
        from h60.tar_stream import TarStreamReader

        noise = (b"ustar" + b"0" * 100 + b"garbage-payload").ljust(512, b"\0")
        real = b"".join(_tar_member(f"{DOMAIN}/images/{uuid.UUID(int=2)}.jpg", _jpeg((9, 9, 9))))
        blob = noise + real + b"\0" * 1024
        seen: list[str] = []

        reader = TarStreamReader(lambda member: True, lambda member, payload: seen.append(member.name))
        for index in range(0, len(blob), 333):
            reader.feed(blob[index : index + 333])
        reader.close()
        self.assertEqual(seen, [f"{DOMAIN}/images/{uuid.UUID(int=2)}.jpg"])

    def test_truncated_archive_is_rejected(self) -> None:
        from h60.tar_stream import TarStreamReader

        blob = b"".join(_tar_member(f"{DOMAIN}/images/{uuid.UUID(int=3)}.jpg", _jpeg((1, 1, 1))))
        reader = TarStreamReader(lambda member: True, lambda member, payload: None)
        reader.feed(blob)
        with self.assertRaises(ValueError):
            reader.close()


class H60PairingRuleTests(unittest.TestCase):
    def test_credential_bearing_intent_is_rejected(self) -> None:
        from h60 import pairs

        with self.assertRaises(ValueError):
            pairs.candidate_from_source_task(
                DOMAIN,
                str(uuid.UUID(int=0)),
                0,
                {
                    "image_id": str(uuid.UUID(int=0)),
                    "domain": DOMAIN,
                    "ordered": True,
                    "intent": "Type the api key into the field",
                    "start_bbox": [0, 0, 10, 10],
                    "end_bbox": [20, 20, 30, 30],
                },
            )

    def test_out_of_bounds_box_is_rejected(self) -> None:
        from h60 import pairs

        candidate = {
            "source_task_index": 0,
            "intent": "drag the handle",
            "ordered": True,
            "start_bbox": [0, 0, 10, 10],
            "end_bbox": [9000, 20, 9500, 50],
            "subtype": None,
        }
        with self.assertRaises(ValueError):
            pairs.geometry_ok(candidate, 1024, 640)

    def test_source_task_must_match_its_own_file_name(self) -> None:
        from h60 import pairs

        with self.assertRaises(ValueError):
            pairs.candidate_from_source_task(
                DOMAIN,
                str(uuid.UUID(int=0)),
                0,
                {
                    "image_id": str(uuid.UUID(int=1)),
                    "domain": DOMAIN,
                    "ordered": True,
                    "intent": "drag the handle",
                    "start_bbox": [0, 0, 10, 10],
                    "end_bbox": [20, 20, 30, 30],
                },
            )


if __name__ == "__main__":
    unittest.main()

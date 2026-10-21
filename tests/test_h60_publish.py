"""H60 publication-path tests.

These tests never touch a real provider account and never upload anything.
They prove the properties that the H60 publication path must hold:

* a shard body is byte-reproducible, so its SHA-256 is a stable resume key;
* the local shard body is released only after the provider has been proven to
  hold exactly the published bytes;
* a failed readback keeps the local body and removes the half-published
  provider dataset, so a retry cannot mistake it for a stored shard;
* an already verified shard is not uploaded twice;
* the provider index is read with pagination, and a provider archive that
  wraps the published tree in one common directory is still verified exactly.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from h60 import ingest, publish


def _body(members: dict[str, bytes]) -> bytes:
    buffer = []
    with zipfile.ZipFile(_Sink(buffer), "w", zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return b"".join(buffer)


class _Sink:
    """Minimal file-like sink so a ZIP can be built entirely in memory."""

    def __init__(self, buffer: list) -> None:
        self.buffer = buffer

    def write(self, data: bytes) -> int:
        self.buffer.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        return None


def _shard_members() -> dict[str, bytes]:
    return {
        "images/slider/a.jpg": b"jpeg-bytes-a",
        "images/slider/b.jpg": b"jpeg-bytes-b",
        "tasks/slider/a.json": b'[{"image_id":"a"}]',
        "labels.jsonl": b'{"id":"row"}\n',
        "manifest.json": b'{"schema":"probe"}\n',
    }


def _write_shard(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    sizes: dict[str, int] = {}
    shas: dict[str, str] = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in _shard_members().items():
            archive.writestr(ingest.member_info(name), payload)
    with zipfile.ZipFile(path) as bundle:
        for info in bundle.infolist():
            sizes[info.filename] = info.file_size
            digest = hashlib.sha256()
            with bundle.open(info) as member:
                for chunk in iter(lambda: member.read(1 << 20), b""):
                    digest.update(chunk)
            shas[info.filename] = digest.hexdigest()
    return sizes, shas


def _manifest() -> dict:
    return {"original_images": 2, "native_drag_actions": 3}


class H60DeterministicShardTests(unittest.TestCase):
    def test_an_identical_shard_body_produces_identical_archive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "a.zip"
            second = Path(temporary) / "b.zip"
            _write_shard(first)
            _write_shard(second)
            self.assertEqual(
                publish.sha256_file(first),
                publish.sha256_file(second),
                "an identical shard body must produce identical archive bytes",
            )


class H60PublishReceiptTests(unittest.TestCase):
    def _publish(self, staging: Path, shard: Path, **overrides):
        calls: dict[str, list] = {"deleted": [], "created": [], "readback": []}
        defaults = {
            "create_private_dataset": lambda upload_dir, ref, title, subtitle: calls["created"].append(ref),
            "wait_until_visible": lambda ref, token: {},
            "verify_private": lambda ref, token: {
                "provider_is_private": True,
                "provider_view_total_bytes": 123,
                "provider_version": 1,
                "anonymous_view_http_status": 403,
            },
            "verify_readback": lambda ref, token, sizes, shas, work: calls["readback"].append(ref) or {
                "provider_readback_members_checked": len(sizes),
                "provider_readback_bytes_mismatched": 0,
            },
            "delete_dataset": lambda ref: calls["deleted"].append(ref),
        }
        defaults.update(overrides)
        with mock.patch.multiple(publish, **defaults):
            receipt = ingest.publish_shard(
                "h60-dragon-train-slider-0000", "slider", shard, staging, "token", _manifest()
            )
        return receipt, calls

    def test_the_local_body_is_released_only_after_a_verified_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            shard = staging / "h60-dragon-train-slider-0000.zip"
            _write_shard(shard)
            receipt, calls = self._publish(staging, shard)
            self.assertTrue(receipt["provider_readback_verified"])
            self.assertTrue(receipt["provider_published"])
            self.assertTrue(receipt["provider_is_private"])
            self.assertEqual(receipt["anonymous_provider_http_status"], 403)
            self.assertEqual(len(calls["readback"]), 1)
            self.assertEqual(calls["deleted"], [])
            self.assertFalse(shard.exists(), "a verified shard must release its local body")

    def test_a_failed_readback_keeps_the_local_body_and_removes_the_provider_dataset(self) -> None:
        def failing_readback(ref, token, sizes, shas, work):
            raise ValueError("provider readback bytes differ for 2 members")

        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            shard = staging / "h60-dragon-train-slider-0000.zip"
            _write_shard(shard)
            with self.assertRaises(ValueError):
                self._publish(staging, shard, verify_readback=failing_readback)
            self.assertTrue(shard.exists(), "a failed verification must keep the local body")
            receipt = json.loads(
                (staging / "shard-receipts" / "slider" / "h60-dragon-train-slider-0000.json").read_text("utf8")
            )
            self.assertFalse(receipt["provider_readback_verified"])
            self.assertTrue(receipt["provider_deleted_after_failure"])
            self.assertIn("provider readback bytes differ", receipt["failure"])

    def test_an_already_verified_shard_is_not_uploaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            shard = staging / "h60-dragon-train-slider-0000.zip"
            _write_shard(shard)
            digest = publish.sha256_file(shard)
            receipt_dir = staging / "shard-receipts" / "slider"
            receipt_dir.mkdir(parents=True)
            (receipt_dir / "h60-dragon-train-slider-0000.json").write_text(
                json.dumps(
                    {
                        "provider_published": True,
                        "provider_readback_verified": True,
                        "local_zip_sha256": digest,
                    }
                ),
                encoding="utf8",
            )
            receipt, calls = self._publish(staging, shard)
            self.assertTrue(receipt["resumed_from_verified_receipt"])
            self.assertEqual(calls["created"], [], "a verified shard must not be uploaded twice")
            self.assertFalse(shard.exists())

    def test_an_uploaded_but_unverified_shard_is_never_uploaded_again(self) -> None:
        """A provider dataset that already holds the body is reused, not rebuilt."""
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            shard = staging / "h60-dragon-train-slider-0000.zip"
            _write_shard(shard)
            calls: dict[str, list] = {"created": [], "deleted": []}
            with mock.patch.multiple(
                publish,
                create_private_dataset=lambda *a, **k: calls["created"].append(a),
                delete_dataset=lambda ref: calls["deleted"].append(ref),
                wait_until_visible=lambda ref, token: {},
                verify_private=lambda ref, token: {
                    "provider_is_private": True,
                    "provider_view_total_bytes": 10 ** 9,
                    "provider_version": 1,
                    "anonymous_view_http_status": 403,
                },
                verify_readback=lambda ref, token, sizes, shas, work: {
                    "provider_readback_members_checked": len(sizes),
                    "provider_readback_bytes_mismatched": 0,
                },
                reusable_private_dataset=lambda ref, token, expected_bytes: True,
            ):
                receipt = ingest.publish_shard(
                    "h60-dragon-train-slider-0000", "slider", shard, staging, "token", _manifest()
                )
            self.assertTrue(receipt["provider_reused_existing_dataset"])
            self.assertEqual(calls["created"], [], "an uploaded body must not be uploaded twice")
            self.assertTrue(receipt["provider_readback_verified"])

    def test_a_failed_readback_keeps_an_uploaded_body_for_the_next_retry(self) -> None:
        def failing_readback(ref, token, sizes, shas, work):
            raise publish.ProviderNotReady("provider did not serve the content")

        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "staging"
            staging.mkdir()
            shard = staging / "h60-dragon-train-slider-0000.zip"
            _write_shard(shard)
            deleted: list[str] = []
            with mock.patch.multiple(
                publish,
                create_private_dataset=lambda *a, **k: None,
                delete_dataset=lambda ref: deleted.append(ref),
                wait_until_visible=lambda ref, token: {},
                verify_private=lambda ref, token: {
                    "provider_is_private": True,
                    "provider_view_total_bytes": 10 ** 9,
                    "provider_version": 1,
                    "anonymous_view_http_status": 403,
                },
                verify_readback=failing_readback,
                reusable_private_dataset=lambda ref, token, expected_bytes: True,
            ):
                with self.assertRaises(publish.ProviderNotReady):
                    ingest.publish_shard(
                        "h60-dragon-train-slider-0000", "slider", shard, staging, "token", _manifest()
                    )
            self.assertEqual(deleted, [], "a correctly uploaded body must not be discarded")
            receipt = json.loads(
                (staging / "shard-receipts" / "slider" / "h60-dragon-train-slider-0000.json").read_text("utf8")
            )
            self.assertTrue(receipt["provider_retained_for_retry"])
            self.assertFalse(receipt["provider_readback_verified"])


class H60ProviderReadbackTests(unittest.TestCase):
    def _provider_archive(self, directory: Path, members: dict[str, bytes]) -> Path:
        path = directory / "provider-dataset.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return path

    def _expected(self) -> tuple[dict[str, int], dict[str, str]]:
        sizes: dict[str, int] = {}
        shas: dict[str, str] = {}
        for name, payload in _shard_members().items():
            sizes[name] = len(payload)
            shas[name] = hashlib.sha256(payload).hexdigest()
        return sizes, shas

    def test_a_provider_archive_with_one_common_directory_is_verified_exactly(self) -> None:
        sizes, shas = self._expected()
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            wrapped = {
                f"expanded/{name}": payload for name, payload in _shard_members().items()
            }
            archive = self._provider_archive(work, wrapped)
            with mock.patch.object(publish, "download_dataset", return_value=archive):
                receipt = publish.verify_readback(
                    "owner/dataset", "token", sizes, shas, work / "readback"
                )
            self.assertEqual(receipt["provider_readback_members_checked"], len(sizes))
            self.assertEqual(receipt["provider_readback_bytes_mismatched"], 0)

    def test_a_member_whose_provider_bytes_differ_is_rejected(self) -> None:
        sizes, shas = self._expected()
        tampered = dict(_shard_members())
        tampered["images/slider/a.jpg"] = b"tampered-bytes"
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            archive = self._provider_archive(work, tampered)
            with mock.patch.object(publish, "download_dataset", return_value=archive):
                with self.assertRaises(ValueError):
                    publish.verify_readback(
                        "owner/dataset", "token", sizes, shas, work / "readback"
                    )

    def test_a_missing_member_is_rejected(self) -> None:
        sizes, shas = self._expected()
        partial = {name: payload for name, payload in _shard_members().items() if name != "labels.jsonl"}
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            archive = self._provider_archive(work, partial)
            with mock.patch.object(publish, "download_dataset", return_value=archive):
                with self.assertRaises(ValueError):
                    publish.verify_readback(
                        "owner/dataset", "token", sizes, shas, work / "readback"
                    )


class H60ProviderIndexTests(unittest.TestCase):
    def test_the_provider_index_is_read_with_pagination(self) -> None:
        pages = [
            {
                "datasetFiles": [{"name": f"images/slider/{i}.jpg", "totalBytes": i} for i in range(200)],
                "nextPageTokenNullable": "page-2",
            },
            {
                "datasetFiles": [{"name": "manifest.json", "totalBytes": 12}],
                "nextPageTokenNullable": None,
            },
        ]
        seen: list[str] = []

        def fake_get(path: str, token: str, accept: str = "application/json", timeout: int = 120):
            seen.append(path)
            return json.dumps(pages[0 if len(seen) == 1 else 1]).encode()

        with mock.patch.object(publish, "_rest_get", side_effect=fake_get):
            files, complete = publish.dataset_files("owner/dataset", "token")
        self.assertTrue(complete)
        self.assertEqual(len(files), 201)
        self.assertIn("manifest.json", files)
        self.assertIn("pageToken=page-2", seen[1])


class H60ProviderCommandTests(unittest.TestCase):
    def test_the_provider_cli_output_is_decoded_as_utf8(self) -> None:
        """The provider CLI prints bytes that are not valid in the console code page."""
        captured: dict = {}

        class _Result:
            returncode = 0
            stdout = "created é中"
            stderr = ""

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command
            return _Result()

        with mock.patch.object(publish.subprocess, "run", side_effect=fake_run):
            result = publish._kaggle("datasets", "create", "-p", "dir")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(captured.get("encoding"), "utf-8")
        self.assertEqual(captured.get("errors"), "replace")

    def test_the_propagation_gate_tolerates_a_not_found_then_serves(self) -> None:
        import urllib.error

        answers = [
            urllib.error.HTTPError("url", 404, "Not Found", None, None),
            urllib.error.HTTPError("url", 403, "Forbidden", None, None),
            {"isPrivate": True, "totalBytes": 10},
        ]

        def fake_get(path: str, token: str) -> dict:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with mock.patch.object(publish, "api_get", side_effect=fake_get):
            with mock.patch.object(publish.time, "sleep"):
                view = publish.wait_until_visible("owner/dataset", "token", timeout=120)
        self.assertTrue(view["isPrivate"])
        self.assertEqual(answers, [])

    def test_the_propagation_gate_gives_up_instead_of_claiming_a_privacy_failure(self) -> None:
        import urllib.error

        def fake_get(path: str, token: str) -> dict:
            raise urllib.error.HTTPError("url", 404, "Not Found", None, None)

        with mock.patch.object(publish, "api_get", side_effect=fake_get):
            with mock.patch.object(publish.time, "sleep"):
                with self.assertRaises(TimeoutError):
                    publish.wait_until_visible("owner/dataset", "token", timeout=0.1)


if __name__ == "__main__":
    unittest.main()

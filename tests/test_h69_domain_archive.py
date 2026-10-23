"""H69 CPU domain transport uses small synthetic source fixtures, no GPU."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from h69 import domain_archive as archive
from h60.source_fetch import REV

COMMIT = "f" * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(root, *, count=2):
    images = root / "images" / "slider"
    ranges = root / "range-receipts"
    images.mkdir(parents=True)
    ranges.mkdir()
    (ranges / "slider-image-0000.receipt.json").write_text('{"range":"verified"}')
    rows = []
    for index in range(count):
        name = f"00000000-0000-0000-0000-{index:012d}"
        payload = ("real-byte-integrity-fixture-" + name).encode()
        (images / (name + ".jpg")).write_bytes(payload)
        rows.append({
            "id": f"test-{index}", "split": "train", "domain": "slider",
            "image_id": name, "image_path": f"images/slider/{name}.jpg",
            "image_sha256": sha(payload), "action_type": "drag",
            "source_task_sha256": "a" * 64,
        })
    labels = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (root / "labels.jsonl").write_bytes(labels)
    receipt = {
        "schema": "hala.h61.source-export.v1",
        "source_domain": "slider", "source_commit_sha": REV,
        "split": "train", "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False, "images_wanted": count,
        "original_images": count, "images_with_original_tasks": count,
        "native_drag_actions": count, "labels_sha256": sha(labels),
        "range_receipts": 1,
    }
    (root / "receipt.json").write_text(json.dumps(receipt))


class SourceShardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.original = self.work / "original"
        source(self.original)
        self.bundle = self.work / "source.hala.bin"
        self.imported = self.work / "restored"

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return archive.pack(self.original, "slider", self.bundle, COMMIT)

    def recover(self, bundle=None, digest=None):
        target = bundle or self.bundle
        return archive.unpack(target, digest or archive.sha(target),
                              self.imported, COMMIT, "slider")

    def test_real_source_contract_roundtrip_and_every_jpeg_sha(self):
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            bundled = self.build()
            self.assertEqual(bundled["images"], 2)
            result = self.recover()
            self.assertEqual(result["status"], "VERIFIED_CPU_SOURCE_RESTORED")
            self.assertEqual(result["actions"], 2)
            self.assertEqual(result["source_revision"], REV)

    def test_archive_is_create_only(self):
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            self.build()
            with self.assertRaises(FileExistsError):
                self.build()

    def test_wrong_archive_sha_does_not_extract(self):
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            self.build()
            with self.assertRaisesRegex(ValueError, "untrusted source archive"):
                self.recover(digest="0" * 64)
            self.assertFalse(self.imported.exists())

    def test_wrong_original_source_revision_fails(self):
        payload = json.loads((self.original / "receipt.json").read_text())
        payload["source_commit_sha"] = "b" * 40
        (self.original / "receipt.json").write_text(json.dumps(payload))
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            with self.assertRaisesRegex(ValueError, "not full, pinned"):
                self.build()

    def test_missing_jpeg_refused(self):
        next((self.original / "images/slider").glob("*.jpg")).unlink()
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                self.build()

    def test_byte_mutation_detected_before_provider_transfer(self):
        img = next((self.original / "images/slider").glob("*.jpg"))
        img.write_bytes(img.read_bytes() + b"tamper")
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            with self.assertRaisesRegex(ValueError, "JPEG byte mismatch"):
                self.build()

    def test_anonymous_dev_record_refused_as_train(self):
        rows = [json.loads(x) for x in (self.original / "labels.jsonl").read_text().splitlines()]
        rows[0]["split"] = "dev"
        labels = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        (self.original / "labels.jsonl").write_bytes(labels)
        receipt = json.loads((self.original / "receipt.json").read_text())
        receipt["labels_sha256"] = sha(labels)
        (self.original / "receipt.json").write_text(json.dumps(receipt))
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            with self.assertRaisesRegex(ValueError, "corrupt original TRAIN row"):
                self.build()

    def test_injected_traversal_member_is_refused(self):
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            self.build()
            with zipfile.ZipFile(self.bundle, "a") as z:
                z.writestr("../outside", "bad")
            with self.assertRaisesRegex(ValueError, "manifest or immutable"):
                self.recover()
            self.assertFalse((self.work / "outside").exists())

    def test_forbid_symlinks(self):
        (self.original / "alias").symlink_to(self.original / "receipt.json")
        with mock.patch.dict(archive.EXPECTED_IMAGES, {"slider": 2}):
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                self.build()


if __name__ == "__main__":
    unittest.main()

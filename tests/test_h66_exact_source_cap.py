"""H66: one HTTP range cannot overfill the screenshot budget.

The real 2026-10-23 CPU export asked for 5 slider images and retained 68
because a 4 MiB range contained multiple complete TAR members. This fake TAR
test reproduces precisely that control flow without downloading source data.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from h61 import source_export as exporter


class FakeRange:
    def __init__(self, domain, start, end, total, receipt_dir, chunk_bytes, label):
        self.label = label
        self.bytes_read = 4 * 1024 * 1024
        self.sha256 = "a" * 64
        self.receipts = [{"label": label}]

    def chunks(self):
        yield b"one HTTP range with 68 complete screenshot members"


class FakeTar:
    counter = 0

    def __init__(self, keep, sink):
        self.keep = keep
        self.sink = sink
        self.image_stream = FakeTar.counter == 0
        FakeTar.counter += 1

    def feed(self, payload):
        for index in range(68):
            key = f"00000000-0000-0000-0000-{index:012d}"
            kind = "images" if self.image_stream else "tasks"
            suffix = "jpg" if self.image_stream else "json"
            member = SimpleNamespace(
                name=f"slider/{kind}/{key}.{suffix}",
                offset=index * 1024,
                size=16,
            )
            if self.keep(member):
                self.sink(member, b"synthetic mock member bytes")


def fake_candidate(*args):
    return {
        "intent": "drag the element", "ordered": True,
        "start_bbox": [10, 10, 20, 20], "end_bbox": [30, 30, 40, 40],
        "subtype": None,
    }


def fake_row(domain, key, candidate, width, height, *args):
    return {
        "id": key, "domain": domain, "ordered": True, "subtype": None,
        "image_width": width, "image_height": height,
        "start_bbox": candidate["start_bbox"],
        "end_bbox": candidate["end_bbox"],
        "completion": {
            "type": "drag", "coordinate_space": "source_image_pixels",
            "ordered": True, "start_point": [15, 15], "end_point": [35, 35],
        },
    }


class ExactCaptureBudgetTests(unittest.TestCase):
    def setUp(self):
        FakeTar.counter = 0

    def export(self, requested):
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(exporter, "RangeReader", FakeRange),
                mock.patch.object(exporter, "TarStreamReader", FakeTar),
                mock.patch.object(exporter, "find_task_region_start",
                                  return_value=(0, "slider/tasks/000.json")),
                mock.patch.object(exporter, "official_lfs_digest",
                                  return_value="f"*64),
                mock.patch.object(exporter.pairs, "is_uuid", return_value=True),
                mock.patch.object(exporter.pairs, "image_tail_ok", return_value=True),
                mock.patch.object(exporter.pairs, "image_digest",
                                  return_value=(1024, 640, "b"*64)),
                mock.patch.object(exporter.pairs, "parse_source_tasks",
                                  return_value=[{}]),
                mock.patch.object(exporter.pairs, "candidate_from_source_task",
                                  side_effect=fake_candidate),
                mock.patch.object(exporter.pairs, "build_row",
                                  side_effect=fake_row),
            ):
                receipt = exporter.export_domain(
                    "slider", 8 * 1024 * 1024, Path(directory) / "slice",
                    images_wanted=requested, chunk_bytes=4 * 1024 * 1024)
                images = list((Path(directory) / "slice/images/slider").glob("*.jpg"))
                return receipt, len(images)

    def test_five_requested_images_means_exactly_five_and_not_sixty_eight(self):
        receipt, files = self.export(5)
        self.assertEqual(receipt["images_wanted"], 5)
        self.assertEqual(receipt["original_images"], 5)
        self.assertEqual(files, 5)
        self.assertEqual(receipt["native_drag_actions"], 5)

    def test_single_image_does_not_inherit_a_batch_of_sixty_eight(self):
        receipt, files = self.export(1)
        self.assertEqual(receipt["original_images"], 1)
        self.assertEqual(files, 1)

    def test_cap_equal_to_chunk_content_keeps_all_real_candidates(self):
        receipt, files = self.export(68)
        self.assertEqual(receipt["original_images"], 68)
        self.assertEqual(files, 68)


if __name__ == "__main__":
    unittest.main()

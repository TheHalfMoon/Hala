"""H348: verified screenshot bytes are the exact bytes decoded by the trainer.

Every fixture is entirely invented; no real or private GUI source is used.
"""
import io
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from training.scripts.train_adapter import (
    check_media, prepare_inputs, verified_media_bytes,
)


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Batch:
    shape = (1,)
    dtype = "int64"

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key != 0:
            raise AssertionError("Only one training item")
        return Vector(self.values)


class Processor:
    def apply_chat_template(self, messages, **_):
        return "prompt" if len(messages) == 1 else "prompt_complete"

    def __call__(self, *, text, images, return_tensors):
        if (return_tensors != "pt" or
                images[0].getpixel((0, 0)) != (240, 20, 10)):
            raise AssertionError("Unverified/replaced pixels reached the processor")
        return SimpleNamespace(input_ids=Batch(
            [10, 11] if text == ["prompt"] else [10, 11, 12, 13]
        ))


class ExactImageSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image_path = self.root / "invented.png"
        self.first = self.make_png((240, 20, 10))
        self.replacement = self.make_png((10, 20, 240))
        self.assertEqual(len(self.first), len(self.replacement))
        self.image_path.write_bytes(self.first)
        self.row = {
            "image_path": "invented.png",
            "image_sha256": sha256(self.first).hexdigest(),
            "prompt": "Invented prompt",
            "completion": '{"type":"observe"}',
        }

    @staticmethod
    def make_png(rgb):
        raw = io.BytesIO()
        Image.new("RGB", (4, 4), rgb).save(raw, "PNG")
        return raw.getvalue()

    def replacement_when_decoder_opens(self, source, *args, **kwargs):
        # Deterministically replace the on-disk original after the code
        # finishes its checksum and immediately before the decoder opens it.
        self.image_path.write_bytes(self.replacement)
        self.assertIsInstance(source, io.BytesIO,
                              "Decoder must consume the verified byte snapshot")
        result = self.original_image_open(source, *args, **kwargs)
        self.assertEqual(result.getpixel((0, 0)), (240, 20, 10))
        return result

    def test_media_header_checks_same_verified_snapshot_after_disk_swap(self):
        self.original_image_open = Image.open
        with patch("PIL.Image.open", side_effect=self.replacement_when_decoder_opens):
            check_media(self.row, self.root)
        self.assertEqual(self.image_path.read_bytes(), self.replacement)

    def test_actual_processor_never_observes_replaced_pixels(self):
        self.original_image_open = Image.open
        fake_torch = SimpleNamespace(tensor=lambda values, dtype: values)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "PIL.Image.open", side_effect=self.replacement_when_decoder_opens
        ):
            inputs, labels = prepare_inputs(self.row, Processor(), self.root, 256)
        self.assertEqual(inputs.input_ids.shape, (1,))
        self.assertEqual(labels[0], [-100, -100, 12, 13])
        self.assertEqual(self.image_path.read_bytes(), self.replacement)

    def test_substituted_image_and_over_budget_bytes_rejected(self):
        self.image_path.write_bytes(self.replacement)
        with self.assertRaisesRegex(ValueError, "digest"):
            verified_media_bytes(self.row, self.root)
        self.image_path.write_bytes(b"0" * 65)
        with patch("training.scripts.train_adapter.MAX_IMAGE_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                verified_media_bytes(self.row, self.root)

    def test_correct_image_snapshot_matches_approved_digest(self):
        self.assertEqual(verified_media_bytes(self.row, self.root), self.first)


if __name__ == "__main__":
    unittest.main()

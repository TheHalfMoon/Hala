"""H349: development training decodes the exact CC0 image bytes it verified.

Invented fixtures only. No actual model inference, GPU, or data admission.
"""
import io
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from PIL import Image

from tests.test_development_training import Processor
from training.development_data import encode_row


class ImageInspectingProcessor(Processor):
    def __call__(self, **kwargs):
        self.seen_pixel = kwargs["images"][0].getpixel((0, 0))
        if self.seen_pixel != (240, 40, 20):
            raise AssertionError("Unverified replacement screenshot reached processor")
        return super().__call__(**kwargs)


class H349DevelopmentByteSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "invented.png"
        self.approved = self.make_png((240, 40, 20))
        self.replacement = self.make_png((20, 40, 240))
        self.path.write_bytes(self.approved)
        self.row = {
            "component": "type", "image": "invented.png",
            "image_sha256": sha256(self.approved).hexdigest(),
            "prompt": "Invented prompt", "completion": '{"type":"observe"}',
        }

    @staticmethod
    def make_png(color):
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), color).save(buffer, "PNG")
        return buffer.getvalue()

    def test_swap_after_verified_read_does_not_change_decoded_pixels(self):
        real_open = Image.open
        seen = []

        def malicious_switch(file):
            self.path.write_bytes(self.replacement)
            self.assertIsInstance(file, io.BytesIO,
                                  "Image reader must receive verified bytes, not path")
            seen.append(True)
            return real_open(file)

        processor = ImageInspectingProcessor()
        _inputs, labels = encode_row(
            self.row, processor, self.root, malicious_switch
        )
        self.assertEqual(seen, [True])
        self.assertEqual(processor.seen_pixel, (240, 40, 20))
        self.assertEqual(self.path.read_bytes(), self.replacement)
        self.assertEqual(labels, [-100, -100, 3])

    def test_substituted_image_digest_refused_before_open(self):
        self.path.write_bytes(self.replacement)
        called = []

        def must_not_open(_):
            called.append(True)
            raise AssertionError("Digest bypass")
        with self.assertRaisesRegex(ValueError, "digest"):
            encode_row(self.row, ImageInspectingProcessor(), self.root,
                       must_not_open)
        self.assertFalse(called)

    def test_mind_component_does_not_open_media(self):
        row = {
            "component": "mind", "image": None, "image_sha256": None,
            "prompt": "Invented mind prompt", "completion": '{"kind":"observe"}',
        }
        _inputs, labels = encode_row(
            row, Processor(), self.root,
            lambda _: self.fail("Mind must stay text only"),
        )
        self.assertEqual(labels, [-100, -100, 3])


if __name__ == "__main__":
    unittest.main()

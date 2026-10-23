"""H71: actual 10k TRAIN media manifest size must fit a bounded metadata gate."""
import json
import tempfile
import unittest
from pathlib import Path

from h68.readiness import _json


class MetadataSizeTests(unittest.TestCase):
    def test_full_train_manifest_over_two_mebibytes_is_valid_only_with_explicit_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "h61_drag_curriculum_manifest.json"
            path.write_text(json.dumps({"schema": "test", "media_padding": "x"*2300000}))
            self.assertGreater(path.stat().st_size, 2*1024*1024)
            with self.assertRaisesRegex(ValueError, "oversized"):
                _json(path)
            actual = _json(path, max_bytes=16*1024*1024)
            self.assertEqual(actual["schema"], "test")

    def test_large_unbounded_manifest_is_still_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "h61_drag_curriculum_manifest.json"
            path.write_text('{"padding":"' + "x"*(16*1024*1024) + '"}')
            with self.assertRaisesRegex(ValueError, "oversized"):
                _json(path, max_bytes=16*1024*1024)

    def test_independent_receipt_limit_remains_two_mebibytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            path.write_text(json.dumps({"padding": "x"*2300000}))
            with self.assertRaisesRegex(ValueError, "oversized"):
                _json(path)


if __name__ == "__main__":
    unittest.main()

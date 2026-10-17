"""H24 prospective archive invariants and exact published source-byte checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h24 import data as h24

ROOT = Path(__file__).resolve().parents[1] / "development/h24-prospective"
PINS = {
    "cases.json": "391d0dc925790558fd5d58f6c0e6348c7c7984b05891fbbd9ae956c9564572de",
    "manifest.json": "5752e9b824b45356d18f5de7c9281d932771e8a2a18c00655506ce8d52dd1dc2",
    "train-mind-prospective.jsonl": "22b37e04bbe2a9281d675922087355c30482efe1126292af4140f8a7f197fd96",
}


class H24ArchiveTests(unittest.TestCase):
    def test_frozen_file_hashes(self):
        for name, sha in PINS.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha)

    def test_canonical_published_dataset(self):
        raw = (ROOT / "cases.json").read_bytes()
        self.assertEqual(raw, h24.canonical(h24.dataset()))
        self.assertEqual(h24.validate(json.loads(raw))["sha256"], PINS["cases.json"])

    def test_exact_train_projection_and_no_test_export(self):
        rows = h24.prospective_rows(h24.dataset())
        expected = b"".join(h24.canonical(row) for row in rows)
        self.assertEqual(expected, (ROOT / "train-mind-prospective.jsonl").read_bytes())
        self.assertEqual(len(rows), 60)
        self.assertFalse({r["family"] for r in rows} &
                         (set(h24.SPLITS["dev"]) | set(h24.SPLITS["test"])))

    def test_manifest_descriptive_truth(self):
        actual = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(actual, h24.validate(h24.dataset()))
        self.assertFalse(actual["type_image_training_qualified"])
        self.assertFalse(actual["native_qualified"])

    def test_case_id_provenance_and_absence_of_external_content(self):
        raw = h24.dataset()
        self.assertEqual(raw["rights"]["source"], "original_procedural_synthetic")
        self.assertFalse(raw["rights"]["third_party_screenshots"])
        self.assertFalse(raw["rights"]["benchmark_trajectories"])
        self.assertTrue(all(case["id"].startswith("h24-") and case["image_qualified"] is False
                            for case in raw["cases"]))

    def test_cli_write_is_create_only_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            self.assertFalse(output.exists())
            from unittest.mock import patch
            with patch("sys.argv", ["h24_data", "--output", str(output)]):
                h24.main()
            self.assertEqual((output / "cases.json").read_bytes(), (ROOT / "cases.json").read_bytes())
            before = hashlib.sha256((output / "cases.json").read_bytes()).hexdigest()
            with patch("sys.argv", ["h24_data", "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    h24.main()
            self.assertEqual(before, hashlib.sha256((output / "cases.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

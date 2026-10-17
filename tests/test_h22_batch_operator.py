"""CPU-only proof of batch operator's source/provenance and archive boundaries."""
import hashlib
import io
import os
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import h22_batch_operator as batch
from tools import generate_h22_batch as notebook


def zipped(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, content in entries:
            z.writestr(name, content)
    raw = buffer.getvalue()
    for name, _ in entries:
        if chr(92) in name:
            raw = raw.replace(name.replace(chr(92), "/").encode(), name.encode())
    return raw


class H22BatchTests(unittest.TestCase):
    def test_historical_sources_are_immutable_pins(self):
        self.assertEqual(len(batch.SOURCE), 40)
        self.assertEqual(len(batch.FREEZE), 64)
        self.assertEqual(batch.MODES, ("fp16", "int8", "nf4"))
        self.assertEqual(len(batch.DEV), 3)
        self.assertEqual(batch.EARLIER_FP16_ARCHIVE,
            "e098a7631708aa50609e56ba7701c01a1c5cc5804b87f9994b6dded181abf471")

    def test_extract_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "safe"
            batch.safe_unpack(zipped([("manifest.json", "{}"), ("images/test.png", "data")]), destination)
            self.assertEqual((destination / "images/test.png").read_text(), "data")
            with self.assertRaises(FileExistsError):
                batch.safe_unpack(zipped([("a", "b")]), destination)

    def test_traversal_and_duplicate_rejected(self):
        for entries in ([("../outside", "bad")], [("a", "x"), ("a", "y")],
                        [("/absolute", "bad")], [("nested\\back", "bad")]):
            if os.name == "nt" and chr(92) in entries[0][0]:
                continue  # Windows ZipInfo normalizes backslashes on archive read
            with self.subTest(entries=entries):
                with tempfile.TemporaryDirectory() as temp:
                    with self.assertRaises(ValueError):
                        batch.safe_unpack(zipped(entries), Path(temp) / "out")

    def test_size_bound_before_files_are_written(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                batch.safe_unpack(zipped([("large", "X" * (8 * 1024**2 + 1))]),
                                  Path(temp) / "out")

    def test_bad_external_commit_fails_before_network(self):
        with self.assertRaises(ValueError):
            batch.public_artifact("main", "some/file", "0" * 64, 10)
        with self.assertRaises(ValueError):
            batch.public_artifact("a" * 40, "../secret", "0" * 64, 10)

    def test_source_has_no_import_time_gpu_or_network(self):
        self.assertNotIn("torch", batch.__dict__)
        self.assertFalse(any(x.startswith("huggingface") for x in batch.__dict__))

    def test_cpu_cannot_start_remote_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                batch.prepare(Path(temp) / "fresh")

    def test_batch_explicitly_excludes_optional_old_torchao(self):
        notebook.generate()
        data = json.loads(notebook.DESTINATION.read_text(encoding="utf-8"))
        package_cell = "".join(data["cells"][3]["source"])
        self.assertIn("%pip uninstall --yes torchao", package_cell)
        self.assertIn("HALA_TORCHAO_OPTIONAL_RUNTIME_ABSENT", package_cell)
        self.assertIn("raise RuntimeError('Incompatible optional torchao present", package_cell)
        self.assertIn("Unqualified optional torchao installation still present",
                      batch.qualify.__code__.co_consts)

    def test_h22_completion_count_schema(self):
        payload = {"status": "measured", "summary": {
            "all": {"episodes": 8, "episodes_complete": 2},
            "en": {"episodes": 4, "episodes_complete": 2},
            "ar": {"episodes": 4, "episodes_complete": 0}}}
        self.assertEqual(batch.completion_counts(payload), {"all": 2, "en": 2, "ar": 0})
        payload["summary"]["en"]["episodes_complete"] = 4
        with self.assertRaises(ValueError):
            batch.completion_counts(payload)

    def test_h22_completion_missing_and_false_zero_rejected(self):
        with self.assertRaises(ValueError):
            batch.completion_counts({"status": "measured", "summary": {"complete": 2}})
        with self.assertRaises(ValueError):
            batch.completion_counts({"status": "blocked", "summary": {}})
        with self.assertRaises(ValueError):
            batch.completion_counts({"status": "measured", "summary": {
                "all": {"episodes": 8, "episodes_complete": False},
                "en": {"episodes": 4, "episodes_complete": 0},
                "ar": {"episodes": 4, "episodes_complete": 0}}})

    def test_notebook_source_is_embedded_exactly(self):
        original = notebook.OPERATOR.read_bytes()
        digest = notebook.generate()
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())
        data = json.loads(notebook.DESTINATION.read_text(encoding="utf-8"))
        self.assertEqual(data["metadata"]["hala_batch_operator_sha256"], digest)
        self.assertEqual(len(data["cells"]), 6)
        setup_source = "".join(data["cells"][-2]["source"])
        self.assertIn("40049660c194f3b8a2541fc7369efebb10e9f94bdac836a2f38fafedd10fa73a", setup_source)
        self.assertIn("ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280", setup_source)
        self.assertIn("HALA_H22_PINNED_FONT_QUALIFIED", setup_source)
        self.assertIn("dpkg-deb", setup_source)
        self.assertIn("--fsys-tarfile", setup_source)
        compile(setup_source, "frozen-font-preflight", "exec")
        self.assertEqual(data["cells"][-1]["cell_type"], "code")
        self.assertIn("NO PUBLIC RESULT CLAIM", "".join(data["cells"][-1]["source"]))
        self.assertEqual(json.loads(notebook.METADATA.read_text())["is_private"], True)


if __name__ == "__main__":
    unittest.main()

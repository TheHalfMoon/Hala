"""H73 offline verifier negative tests on tiny synthetic archives, never raw H71."""
import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from h73.kaggle_cpu_admission import qualify

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def bundle(directory, *, dev_row=False, corrupt_media=False, duplicate=False):
    family = "h61_firstparty_bilingual_drag_train"
    png = [b"safe-synthetic-pixels-one", b"safe-synthetic-pixels-two"]
    paths = ["firstparty/png/scene1.png", "firstparty/png/scene2.png"]
    rows = [
        {"id": f"r{i}", "image_path": paths[i],
         "image_sha256": sha(png[i]), "split": ("dev" if dev_row and i == 0 else "train"),
         "family": family, "language": ["ar", "en"][i],
         "capability": ["sheet/ordered", "slider/ordered"][i]}
        for i in range(2)
    ]
    label_bytes = "".join(json.dumps(row) + "\n" for row in rows).encode("utf8")
    manifest = {
        "schema": "hala.training.manifest.v1",
        "source_schema": "hala.h61.native-drag-curriculum.v1",
        "split": "train", "train_rows": 2, "unique_source_images": 2,
        "dataset_sha256": sha(label_bytes),
        "language_counts": {"ar": 1, "en": 1},
        "training_executed": False,
        "model_improvement_qualified": False,
        "no_benchmark_eval_source": True,
        "family_targets": {family: 2},
        "selected_by_family": {family: 2},
        "capability_counts": {"sheet/ordered": 1, "slider/ordered": 1},
        "media": [{"path": p, "sha256": sha(blob)} for p, blob in zip(paths, png)],
    }
    content = {
        "export/h61_drag_curriculum.jsonl": label_bytes,
        "export/h61_drag_curriculum_manifest.json": json.dumps(manifest).encode(),
        "public-dev-baseline-receipt.json": b'{"baseline":"frozen"}',
        **{"export/" + path: data for path, data in zip(paths, png)},
    }
    index = {
        "schema": "hala.h67.cpu-stage.v1", "git_sha": "a" * 40,
        "source_split": "train", "dataset_sha256": sha(label_bytes),
        "row_count": 2, "media_count": 2,
        "files": {name: {"sha256": sha(data), "size": len(data)}
                  for name, data in content.items()},
    }
    if corrupt_media:
        content["export/" + paths[0]] = b"modified-media"
        # Keep frozen original inventory to catch tampering.
    archive = Path(directory) / "small.hala.bin"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in content.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                z.writestr("export/" + paths[0], b"duplicate member")
        z.writestr("h67-stage.json", json.dumps(index))
    expected = {"sha256": sha(archive.read_bytes()), "bytes": archive.stat().st_size,
                "git_sha": "a"*40, "dataset_sha256": sha(label_bytes),
                "files": len(content), "rows": 2,
                "language_counts": {"ar": 1, "en": 1},
                "firstparty_rows": 2}
    return archive, expected

class OfflineAdmissionTests(unittest.TestCase):
    def test_synthetic_cpu_proof_with_no_model_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, expected = bundle(folder)
            proof = qualify(archive, expected)
            self.assertEqual(proof["rows_verified"], 2)
            self.assertEqual(proof["members_sha_verified"], 5)
            self.assertFalse(proof["gpu_allocated"])
            self.assertFalse(proof["training_executed"])
            self.assertFalse(proof["provider_visibility_verified_by_owner"])

    def test_false_archive_digest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, expected = bundle(folder)
            expected["sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "archive SHA256 mismatch"):
                qualify(archive, expected)

    def test_modified_screenshot_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, expected = bundle(folder, corrupt_media=True)
            with self.assertRaisesRegex(ValueError, "member byte proof failed"):
                qualify(archive, expected)

    def test_dev_row_is_never_train(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, expected = bundle(folder, dev_row=True)
            with self.assertRaisesRegex(ValueError, "non-TRAIN"):
                qualify(archive, expected)

    def test_duplicate_members_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, expected = bundle(folder, duplicate=True)
            expected["files"] += 1
            with self.assertRaisesRegex(ValueError, "duplication"):
                qualify(archive, expected)

    def test_nonexistent_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "absent"):
                qualify(Path(folder) / "missing.bin")

if __name__ == "__main__":
    unittest.main()

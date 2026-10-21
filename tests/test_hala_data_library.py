"""Offline integrity tests for Hala's first PRIVATE Kaggle training library."""
import hashlib, json, tempfile, unittest, zipfile
from pathlib import Path
from research.data_library.build_kaggle_bundle import ROOT, build, DATASET

CATALOG=ROOT/"research/data_library/catalog.json"
STAGES=("discovered","upstream_pinned","quality_reviewed","stored_kaggle",
        "training_completed","independent_evaluation_completed")


class HalaDataLibraryTests(unittest.TestCase):
    def test_registered_actual_private_kaggle_storage_and_backed_sha(self):
        v=json.loads(CATALOG.read_text("utf-8"))
        self.assertEqual(v["schema"],"hala.training-data-library-progress.v1")
        storage=v["storage"]
        self.assertEqual(storage["kaggle_dataset"],DATASET)
        self.assertTrue(storage["kaggle_private_verified"])
        self.assertTrue(storage["kaggle_metadata_readback_verified"])
        original=ROOT/"h50/assets/authoring_sft.jsonl"
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         storage["remote_training_jsonl_sha256"])
        self.assertEqual(len(v["datasets"]),14)

    def test_no_other_data_is_claimed_uploaded_or_trained(self):
        v=json.loads(CATALOG.read_text("utf-8"))
        for row in v["datasets"]:
            self.assertEqual(set(row["progress"]),set(STAGES))
            self.assertTrue(row["progress"]["discovered"])
            self.assertFalse(row["progress"]["training_completed"])
            self.assertFalse(row["progress"]["independent_evaluation_completed"])
            self.assertIsNone(row["trained_model_checkpoint"])
            if row["id"] not in ("hala-h50-original","hala-h52-original-task-factory"):
                self.assertFalse(row["progress"]["stored_kaggle"])
                self.assertFalse(row["remote_data_imported"])
        self.assertEqual(sum(x["progress"]["stored_kaggle"] for x in v["datasets"]),2)

    def test_builder_is_train_only_and_create_only(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/"kaggle"
            build(dest)
            with zipfile.ZipFile(dest/"hala-h50-train-original.zip") as z:
                names=z.namelist()
                self.assertEqual(len(names),68)
                self.assertEqual(sum(n.endswith(".png") for n in names),64)
                self.assertTrue(any(n.endswith("/authoring_sft.jsonl") for n in names))
                self.assertFalse(any("evaluator_only" in n or "dev_public" in n or
                                     "hcompany" in n.lower() for n in names))
                self.assertIsNone(z.testzip())
                raw=z.read("h50-train/authoring_sft.jsonl")
                self.assertEqual(raw,(ROOT/"h50/assets/authoring_sft.jsonl").read_bytes())
            info=json.loads((dest/"library-manifest.json").read_text("utf-8"))
            self.assertTrue(info["private"])
            self.assertEqual(info["unique_screenshots"],64)
            self.assertEqual(info["supervision_rows"],384)
            self.assertFalse(info["external_competitor_data_included"])
            self.assertFalse(info["evaluation_material_included"])
            self.assertEqual(info["archive_sha256"],hashlib.sha256(
                (dest/"hala-h50-train-original.zip").read_bytes()).hexdigest())
            # DEFLATE can produce different ZIP bytes on Linux vs Windows.
            # The actual uploaded Windows archive has its own remote receipt;
            # never confuse its digest with this independently rebuilt ZIP.
            self.assertEqual(info["files"]["authoring_sft.jsonl"],json.loads(
                CATALOG.read_text("utf-8"))["storage"]["remote_training_jsonl_sha256"])
            for item in json.loads((ROOT/"h50/assets/sft_manifest.json").read_text("utf-8"))["media"]:
                self.assertEqual(info["files"][item["path"]],item["sha256"])
            with self.assertRaises(FileExistsError):
                build(dest)

if __name__=="__main__":
    unittest.main()

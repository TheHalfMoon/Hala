"""H73 owner-private CPU Kaggle admission: no GPU, secrets or raw-data outputs."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from h73 import kaggle_h73_cpu as h


class AdmissionTests(unittest.TestCase):
    def test_kaggle_kernel_metadata_is_private_cpu_and_exact_source(self):
        metadata = json.loads((Path(__file__).resolve().parents[1]/"h73/kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"],"abdulazizshehri/hala-h73-cpu-train-source-admission")
        self.assertIs(metadata["is_private"],True)
        self.assertIs(metadata["enable_gpu"],False)
        self.assertEqual(metadata["dataset_sources"],[h.DATASET_REF])
        self.assertEqual(metadata["code_file"],"kaggle_h73_cpu.py")

    def test_missing_or_mismatched_private_source_never_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder)/"stage.bin"
            with self.assertRaisesRegex(ValueError,"missing"):
                h.verify_input(bad)
            bad.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError,"wrong length"):
                h.verify_input(bad)
            with mock.patch.object(h,"ARCHIVE_BYTES",len(bad.read_bytes())):
                with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                    h.verify_input(bad)

    def test_kaggle_2026_owner_mount_locates_only_exact_train_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            target=root/"datasets"/"abdulazizshehri"/"hala-h71-drag-train-10000-private"
            target.mkdir(parents=True)
            expected=target/"hala-h71-train-10000.hala.bin"
            expected.write_bytes(b"small-test-payload")
            self.assertEqual(h.locate_attached_train(root),expected)
            expected.unlink()
            with self.assertRaisesRegex(ValueError,"not mounted"):
                h.locate_attached_train(root)

    def test_exact_file_sha_attests_input_without_model_download(self):
        with tempfile.TemporaryDirectory() as folder:
            blob=Path(folder)/"stage.bin"
            blob.write_bytes(b"authentic-verified-source")
            digest=hashlib.sha256(blob.read_bytes()).hexdigest()
            with mock.patch.object(h,"ARCHIVE_BYTES",blob.stat().st_size),mock.patch.object(h,"ARCHIVE_SHA",digest):
                result=h.verify_input(blob)
            self.assertEqual(result["archive_sha256"],digest)
            self.assertEqual(result["dataset_ref"],h.DATASET_REF)

    def test_cpu_only_rejects_kaggle_gpu_if_visible(self):
        with mock.patch.dict(h.os.environ,{"KAGGLE_KERNEL_RUN_TYPE":"Batch"},clear=True),mock.patch.object(h.shutil,"which",return_value="/usr/bin/nvidia-smi"),mock.patch.object(h.subprocess,"run",return_value=mock.Mock(returncode=0,stdout="GPU 0: NVIDIA Tesla T4")):
            with self.assertRaisesRegex(ValueError,"GPU detected"):
                h.require_cpu()

    def test_cpu_notebook_never_accepts_local_host(self):
        with mock.patch.dict(h.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"actual Kaggle CPU"):
                h.require_cpu()

    def test_incorrect_h67_rederivation_never_counts(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(h,"fetch_pinned_source",return_value=h.H71_CODE_SHA):
            from h67 import stage
            with mock.patch.object(stage,"restore_stage",return_value={
                "status":"RESTORED_TRAIN_STAGE","sha256":h.ARCHIVE_SHA,
                "dataset_sha256":"0"*64,"files":h.EXPECTED_MEMBERS}):
                with self.assertRaisesRegex(ValueError,"integrity admission"):
                    h.verify_source(Path(folder),Path(folder)/"archive")

    def test_small_atomic_receipt_never_contains_raw_training_data(self):
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(h,"OUTPUT",Path(folder)/"receipt.json"):
            h.write_receipt({"status":"VERIFIED","data":{"bytes":123}})
            self.assertEqual(json.loads((Path(folder)/"receipt.json").read_text())["status"],"VERIFIED")
            self.assertFalse((Path(folder)/".receipt.json.tmp").exists())


if __name__=="__main__":
    unittest.main()

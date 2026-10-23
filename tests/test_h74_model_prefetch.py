"""H74 prefetch never accepts changed weights, wrong commit or paid runtime."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h74 import remote_prefetch as p


class PinnedModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache=Path(self.temp.name)/"hub"
        self.snapshot=self.cache/"models--Qwen--Qwen3-VL-2B-Instruct"/"snapshots"/p.MODEL_REVISION
        self.snapshot.mkdir(parents=True)
        (self.snapshot/"config.json").write_text('{"model_type":"qwen3_vl"}')

    def test_metadata_is_pinned_to_exact_revision(self):
        with mock.patch.object(p,"CACHE",self.cache):
            result=p.verify_snapshot(self.snapshot,full=False)
        self.assertEqual(result["revision"],p.MODEL_REVISION)
        self.assertFalse(result["model_weights_verified"])
        self.assertEqual(result["model_type"],"qwen3_vl")

    def test_different_revision_and_escape_refused(self):
        with mock.patch.object(p,"CACHE",self.cache):
            with self.assertRaisesRegex(ValueError,"revision"):
                p.verify_snapshot(self.snapshot.parent/"wrong",full=False)
        with mock.patch.object(p,"CACHE",self.cache):
            with self.assertRaisesRegex(ValueError,"cache root"):
                p.verify_snapshot(Path(self.temp.name),full=False)

    def test_missing_full_weights_never_qualified(self):
        with mock.patch.object(p,"CACHE",self.cache):
            with self.assertRaisesRegex(ValueError,"weights not present"):
                p.verify_snapshot(self.snapshot,full=True)

    def test_full_weights_digest_failure_refuses_even_valid_metadata(self):
        binary=self.snapshot/"model.safetensors"
        binary.write_bytes(b"fake-model-weights")
        # Patch size alone, not content hash. The full digest must still
        # reject a corrupted artifact even if its declared size looks large.
        real=p.Path.stat
        def stat(path,*args,**kwargs):
            value=real(path,*args,**kwargs)
            if Path(path)==binary:
                from unittest.mock import Mock
                fake=Mock()
                fake.st_size=4*1024**3
                return fake
            return value
        with mock.patch.object(p,"CACHE",self.cache),mock.patch.object(p.Path,"stat",stat):
            with self.assertRaisesRegex(ValueError,"SHA256 differs"):
                p.verify_snapshot(self.snapshot,full=True)

    def test_explicit_free_colab_evidence_required(self):
        with mock.patch.dict(p.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"real Colab"):
                p.guard_free_colab()
        with mock.patch.object(p.Path,"is_dir",return_value=True),mock.patch.dict(
            p.os.environ,{"COLAB_RELEASE_TAG":"abc"},clear=True):
            with self.assertRaisesRegex(ValueError,"free non-paid"):
                p.guard_free_colab()

    def test_receipt_emits_only_small_model_identity(self):
        receipt=Path(self.temp.name)/"receipt.json"
        with mock.patch.object(p,"RECEIPT",receipt):
            p.write_receipt({"status":"H74_METADATA_VERIFIED","gpu_training_executed":False})
        self.assertFalse(json.loads(receipt.read_text())["gpu_training_executed"])
        self.assertFalse(receipt.with_name("."+receipt.name+".tmp").exists())


if __name__=="__main__":
    unittest.main()

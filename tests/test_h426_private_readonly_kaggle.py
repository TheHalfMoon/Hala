"""Pure CPU tests for private-only read-only synthetic baseline packaging."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "h426_private_builder",ROOT/"h426/build_private_readonly_kaggle.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class H426OwnerPrivateReadOnlyBundleTest(unittest.TestCase):
    def test_source_only_deterministic_bounded_bundle(self):
        first=module.source_bundle()
        second=module.source_bundle()
        self.assertEqual(first,second)
        self.assertLess(len(first),module.MAX_BUNDLE_BYTES)
        with zipfile.ZipFile(io.BytesIO(first)) as zipped:
            self.assertEqual(tuple(zipped.namelist()),module.PARTS)
            for part in zipped.namelist():
                self.assertTrue(part.endswith(".py"))
                body=zipped.read(part)
                self.assertNotIn(b"-----BEGIN PRIVATE KEY",body)
                self.assertFalse(part.endswith(".safetensors"))
                self.assertLess(len(body),module.MAX_BUNDLE_BYTES)

    def test_kaggle_private_metadata_and_no_training(self):
        metadata=module.metadata()
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(metadata["dataset_sources"],[module.PARENT_DATASET])
        self.assertEqual(metadata["id"],module.KERNEL_ID)
        self.assertFalse(metadata["enable_tpu"])
        self.assertEqual(metadata["model_sources"],[])

    def test_generated_bundle_is_parseable_self_verifying(self):
        code=module.script(module.source_bundle())
        ast.parse(code)
        self.assertIn("source bundle SHA drift",code)
        self.assertIn("runpy.run_path",code)
        self.assertIn("h426/kaggle_h425_readonly_synthetic.py",code)
        self.assertNotIn("h425_original_trained_adapter.safetensors",code)

    def test_never_create_private_kaggle_work_inside_repository(self):
        with self.assertRaises(ValueError):
            module.build(ROOT/"private-kaggle-upload",ROOT)

    def test_runner_does_not_create_optimizer_or_train(self):
        runner=(ROOT/"h426/kaggle_h425_readonly_synthetic.py").read_text("utf8")
        self.assertIn("h425_true_trained_child",runner)
        self.assertIn("trained_receipt",runner)
        self.assertIn("evaluate_already_loaded_model",runner)
        self.assertIn("H425_WEIGHT_SHA",runner)
        self.assertIn("source_rights_review_qualified",runner)
        for forbidden in ("model.train(","optimizer.step(","trainer.train(","model.save_pretrained("):
            self.assertNotIn(forbidden,runner)

if __name__=="__main__":
    unittest.main()

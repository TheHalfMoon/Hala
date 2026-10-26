"""Free Kaggle-T4 private H140 training mount and source isolation guards."""
import json
import tempfile
import unittest
from pathlib import Path

from h140.train_kaggle.kaggle_private_train import (
    APPROVED_MOUNTS,SOURCE_SHA,source_mounts
)


class H140FreePrivateGPUOnlyContracts(unittest.TestCase):
    def test_exact_two_frozen_owner_sources_and_no_blind(self):
        self.assertEqual(APPROVED_MOUNTS,{
            "hala-h140-private-train256-original",
            "hala-h140-private-dev64-original"})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for n in APPROVED_MOUNTS:
                (p/n).mkdir()
            source_mounts(p)
            # Additional known system mount is harmless; dataset admission
            # still requires exactly TRAIN256/DEV64 H140 manifests elsewhere.
            (p/"__kaggle-system").mkdir()
            source_mounts(p)
            (p/"private-owner-h140-blind-original").mkdir()
            with self.assertRaisesRegex(ValueError,"not permitted"):
                source_mounts(p)
            (p/"private-owner-h140-blind-original").rmdir()
            (p/next(iter(APPROVED_MOUNTS))).rmdir()
            with self.assertRaisesRegex(ValueError,"missing"):
                source_mounts(p)

    def test_kernel_is_private_noncommercial_free_t4_metadata(self):
        root=Path(__file__).parent.parent/"h140"/"train_kaggle"
        data=json.loads((root/"kernel-metadata.json").read_text("utf8"))
        self.assertTrue(data["is_private"])
        self.assertTrue(data["enable_gpu"])
        self.assertEqual(data["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(len(data["dataset_sources"]),2)
        self.assertEqual(set(data["dataset_sources"]),
                         {"abdulazizshehri/"+x for x in APPROVED_MOUNTS})
        self.assertEqual(len(SOURCE_SHA),40)
        self.assertNotIn("blind",data["id"])


if __name__=="__main__":
    unittest.main()

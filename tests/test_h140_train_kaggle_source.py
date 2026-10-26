"""Free Kaggle-T4 private H140 training mount and source isolation guards."""
import json
import tempfile
import unittest
from pathlib import Path

from h140.train_kaggle.kaggle_private_train import (
    APPROVED_MOUNTS,SOURCE_SHA,source_mounts
)


class H140FreePrivateGPUOnlyContracts(unittest.TestCase):
    def test_source_roles_identified_without_relying_on_kaggle_mount_name(self):
        self.assertEqual(APPROVED_MOUNTS,{
            "hala-h140-private-train256-original",
            "hala-h140-private-dev64-original"})
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            schema="hala.h140.original-app-multi-geometry-bilingual-source.v1"
            train=root/"kaggle-auto-nested-train-source"
            dev=root/"other-worker-dev-source"
            train.mkdir();dev.mkdir()
            def write(role,folder):
                (folder/"manifest.json").write_text(json.dumps({
                    "schema":schema,"split":role,"private":True}),encoding="utf8")
            write("train",train);write("dev",dev)
            self.assertEqual(source_mounts(root),{"train":1,"dev":1})
            (root/"__kaggle-system").mkdir()
            self.assertEqual(source_mounts(root),{"train":1,"dev":1})
            forbidden=root/"some-ood-mount";forbidden.mkdir()
            write("blind_test",forbidden)
            with self.assertRaisesRegex(ValueError,"forbidden"):
                source_mounts(root)
            (forbidden/"manifest.json").unlink()
            write("train",forbidden)
            with self.assertRaisesRegex(ValueError,"unavailable"):
                source_mounts(root)
            (forbidden/"manifest.json").unlink()
            (dev/"manifest.json").unlink()
            with self.assertRaisesRegex(ValueError,"unavailable"):
                source_mounts(root)

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

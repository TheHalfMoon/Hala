"""Kaggle arbitrary mount-depth manifest admission; sealed holdout forbidden."""
import json
import tempfile
import unittest
from pathlib import Path
from h134.kaggle_private_train import identify_owner_splits

class H134PrivateAdmission(unittest.TestCase):
    def test_nested_kaggle_mount_and_no_sealed_holdout(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            for split in ("train","dev"):
                folder=root/"some-kaggle-owner"/"private"/split
                folder.mkdir(parents=True)
                (folder/"manifest.json").write_text(json.dumps({
                   "schema":"hala.h134.original-chromium-start-end-source.v1",
                   "split":split}),"utf8")
            self.assertEqual(identify_owner_splits(root),{"train":1,"dev":1})
            folder=root/"other"/"sealed"
            folder.mkdir(parents=True)
            (folder/"manifest.json").write_text(json.dumps({
                 "schema":"hala.h134.original-chromium-start-end-source.v1",
                 "split":"sealed_test"}),"utf8")
            with self.assertRaises(ValueError):
                identify_owner_splits(root)
            (folder/"manifest.json").unlink()
            (root/"some-kaggle-owner"/"private"/"dev"/"manifest.json").unlink()
            with self.assertRaises(ValueError):
                identify_owner_splits(root)
if __name__=="__main__":
    unittest.main()

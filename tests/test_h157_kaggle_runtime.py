"""H157 Kaggle single-private-mount and reproducible runtime gates."""
import json
import tempfile
import unittest
from pathlib import Path

from h157.kaggle_train.kaggle_private_train import (
    SOURCE_COMMIT,locate,admit_only_one_mount
)


class H157FreePrivateT4Admission(unittest.TestCase):
    def test_owner_private_source_and_model_can_be_zip_or_directory(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            private=root/"some-dynamic-dataset"
            private.mkdir()
            a=private/"h157-original-source"
            b=private/"h156-trained-typed-action-lora"
            a.mkdir()
            b.mkdir()
            self.assertEqual(locate(root,"h157-original-source",kind="source"),a)
            self.assertEqual(locate(root,"h156-trained-typed-action-lora",kind="adapter"),b)
            admit_only_one_mount(root,a,b)
            (root/"unauthorized-blind-dataset").mkdir()
            with self.assertRaises(ValueError):
                admit_only_one_mount(root,a,b)

    def test_unrelated_or_missing_source_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            with self.assertRaises(ValueError):
                locate(root,"h157-original-source",kind="source")
            self.assertEqual(len(SOURCE_COMMIT),40)

    def test_cannot_mount_source_and_adapter_in_different_private_roots(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            a=root/"first"
            b=root/"second"
            a.mkdir()
            b.mkdir()
            with self.assertRaises(ValueError):
                admit_only_one_mount(root,a,b)


if __name__=="__main__":
    unittest.main()

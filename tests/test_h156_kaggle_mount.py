"""H156 Kaggle ZIP auto-extraction regression, CPU-only."""
import json
import tempfile
import unittest
from pathlib import Path

from h156.train_gpu import FolderMedia,open_media
from h156.kaggle_train.kaggle_private_train import find_source,admit_single_private_mount


class H156KaggleMountTests(unittest.TestCase):
    def test_zip_expanded_directory_can_be_addressed_by_real_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"h156-source-images"
            (root/"images").mkdir(parents=True)
            (root/"manifest.json").write_text("{}",encoding="utf8")
            (root/"images"/"a.png").write_bytes(b"png")
            self.assertEqual(find_source(Path(tmp),"h156-source-images"),root)
            with open_media(root) as media:
                self.assertEqual(set(media.namelist()),
                                 {"manifest.json","images/a.png"})
                self.assertEqual(media.read("images/a.png"),b"png")
                for attack in ("../outside","/secret","a\\b"):
                    with self.assertRaises(ValueError):
                        media.read(attack)

    def test_duplicate_source_mount_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"h156-source-images").mkdir()
            (root/"h156-source-images.zip").write_bytes(b"dummy")
            with self.assertRaises(ValueError):
                find_source(root,"h156-source-images")

    def test_runtime_mount_slug_may_change_but_never_contain_extra_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"kaggle-input"
            mount=root/"mounted-dataset-random-slug"
            mount.mkdir(parents=True)
            source=(mount/"original_images").mkdir()
            adapter=(mount/"original_adapter").mkdir()
            self.assertEqual(admit_single_private_mount(
                root,mount/"original_images",mount/"original_adapter"),mount)
            other=root/"unexpected-extra-dataset"
            other.mkdir()
            with self.assertRaises(ValueError):
                admit_single_private_mount(
                    root,mount/"original_images",mount/"original_adapter")
            other.rmdir()
            outside=Path(tmp)/"outside-adapter"
            outside.mkdir()
            with self.assertRaises(ValueError):
                admit_single_private_mount(root,mount/"original_images",outside)


if __name__=="__main__":
    unittest.main()

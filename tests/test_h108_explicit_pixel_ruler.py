"""H108 original source vs explicit position ruler image preprocessing contracts."""
import hashlib
import tempfile
from pathlib import Path
import unittest

from PIL import Image
from h108.ruler import position_ruler,annotated_prompt,GRID_STEP
from h108.eval import frozen_source,variants

class H108RulerContracts(unittest.TestCase):
    def test_exact_original_frame_and_no_original_pixel_mutation(self):
        photo=Image.new("RGB",(1024,640),(90,123,157))
        before=hashlib.sha256(photo.tobytes()).hexdigest()
        overlaid=position_ruler(photo)
        self.assertEqual(overlaid.size,photo.size)
        self.assertEqual(photo.mode,"RGB")
        self.assertEqual(before,hashlib.sha256(photo.tobytes()).hexdigest())
        self.assertNotEqual(photo.tobytes(),overlaid.tobytes())
    def test_ruler_determinism(self):
        source=Image.new("RGB",(1024,640),(20,30,40))
        first=position_ruler(source)
        second=position_ruler(source)
        self.assertEqual(first.tobytes(),second.tobytes())
        self.assertEqual(GRID_STEP,64)
    def test_wrong_dimensions_refused_no_implicit_source_warp(self):
        with self.assertRaises(ValueError):position_ruler(Image.new("RGB",(495,640)))
        with self.assertRaises(ValueError):position_ruler(Image.new("RGB",(1024,480)))
    def test_unchanged_original_prompt_retained_and_bbox_not_exposed(self):
        prompt="Locate the requested slider in the displayed UI."
        augmented=annotated_prompt(prompt)
        self.assertTrue(augmented.startswith(prompt))
        self.assertIn("original image pixel coordinates",augmented)
        self.assertNotIn("start_bbox",augmented)
        with self.assertRaises(ValueError):annotated_prompt("")
    def test_fresh_original_indices_36_43_only(self):
        scenes=[{"id":str(i)} for i in range(44)]
        chosen=frozen_source(scenes)
        self.assertEqual([x["id"] for x in chosen],[str(i) for i in range(36,44)])
        with self.assertRaises(ValueError):frozen_source(scenes[:-1])
    def test_variant_does_not_edit_source_identity_or_source_task(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p)
            image=root/"original.png"
            Image.new("RGB",(1024,640),(100,120,140)).save(image)
            before=hashlib.sha256(image.read_bytes()).hexdigest()
            original={"id":"private-synthetic","domain":"slider","image_width":1024,
                      "image_height":640,"image_path":str(image),
                      "image_sha256":before,
                      "prompt":"Original source GUI drag instruction.",
                      "task":{"start_bbox":[10,20,30,40],
                              "end_bbox":[510,510,550,550],"ordered":True}}
            untouched,augmented=variants([original],root/"augmented")
            self.assertEqual(untouched[0],original)
            self.assertEqual(augmented[0]["task"],original["task"])
            self.assertEqual(augmented[0]["id"],original["id"])
            self.assertNotEqual(augmented[0]["image_sha256"],original["image_sha256"])
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(),before)
            with Image.open(augmented[0]["image_path"]) as im:
                self.assertEqual(im.size,(1024,640))
            with self.assertRaises(FileExistsError):variants([original],root/"augmented")
if __name__=="__main__":unittest.main()

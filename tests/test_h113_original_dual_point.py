"""H113 original TRAIN-only dual-point relative grounding targets."""
import json,unittest
from copy import deepcopy
from h61.prompt_contract import drag_prompt,source_body
from h113.dual_point import (derive_two_independent_points,source_pixel_to_relative,
    relative_to_source_pixel,point_prompt,admit_original_source)

def fixtures(w=1024,h=640):
    original={"id":"original-synthetic-TRAIN-only","split":"train",
        "action_type":"drag","image_path":"original/train/example.png",
        "image_sha256":"a"*64,"image_width":w,"image_height":h,
        "language":"ar","family":"original_test_family","ordered":True,
        "intent":"Drag the source control to its destination.",
        "start_bbox":[250,250,290,290],"end_bbox":[700,410,740,450]}
    original["completion"]={"type":"drag","coordinate_space":"source_image_pixels",
        "ordered":True,"start_point":[270,270],"end_point":[720,430]}
    row={"id":original["id"],"split":"train","language":"ar",
        "family":original["family"],"image_path":original["image_path"],
        "image_sha256":original["image_sha256"],
        "prompt":drag_prompt(source_body(original["intent"],True),w,h,True),
        "completion":json.dumps(original["completion"],sort_keys=True,separators=(",",":"))}
    return row,original

class TestH113(unittest.TestCase):
    def test_two_source_train_points_source_sha_and_bbox_roundtrip(self):
        row,source=fixtures()
        start,end=derive_two_independent_points(row,source)
        self.assertEqual(start["endpoint_role"],"start")
        self.assertEqual(end["endpoint_role"],"end")
        self.assertEqual(start["split"],end["split"])
        self.assertEqual(start["split"],"train")
        self.assertEqual(start["image_sha256"],end["image_sha256"])
        self.assertEqual(start["image_path"],row["image_path"])
        self.assertNotEqual(start["completion"],end["completion"])
        for output in (start,end):
            result=json.loads(output["completion"])
            self.assertEqual(result["coordinate_space"],"normalized_0_1000")
            point=relative_to_source_pixel(result["point"],1024,640)
            box=source[output["endpoint_role"]+"_bbox"]
            self.assertTrue(box[0]<=point[0]<=box[2])
            self.assertTrue(box[1]<=point[1]<=box[3])
            self.assertNotIn("bbox",output["prompt"])
            self.assertNotIn("source_image_pixels",output["prompt"])

    def test_any_eval_or_changed_immutable_train_label_fails(self):
        row,src=fixtures()
        row["split"]="eval"
        with self.assertRaises(ValueError):derive_two_independent_points(row,src)
        row,src=fixtures();src["split"]="eval"
        with self.assertRaises(ValueError):derive_two_independent_points(row,src)
        row,src=fixtures();src["image_sha256"]="b"*64
        with self.assertRaises(ValueError):derive_two_independent_points(row,src)
        row,src=fixtures();row["prompt"]="altered prompt"
        with self.assertRaises(ValueError):derive_two_independent_points(row,src)
        row,src=fixtures();row["completion"]=row["completion"].replace(
            "source_image_pixels","normalized_0_1000")
        with self.assertRaises(ValueError):derive_two_independent_points(row,src)

    def test_original_noncenter_source_point_inside_bbox_must_be_preserved(self):
        row,src=fixtures()
        src["completion"]["start_point"]=[257,263]
        row["completion"]=json.dumps(src["completion"],sort_keys=True,separators=(",",":"))
        start,end=derive_two_independent_points(row,src)
        point=json.loads(start["completion"])["point"]
        reconstructed=relative_to_source_pixel(point,1024,640)
        self.assertTrue(250<=reconstructed[0]<=290 and 250<=reconstructed[1]<=290)
        self.assertNotEqual(src["completion"]["start_point"],[270,270])
        self.assertEqual(end["endpoint_role"],"end")

    def test_original_point_outside_original_target_box_rejected(self):
        row,src=fixtures()
        src["completion"]["start_point"]=[10,10]
        row["completion"]=json.dumps(src["completion"],sort_keys=True,separators=(",",":"))
        with self.assertRaisesRegex(ValueError,"outside original verified target box"):
            derive_two_independent_points(row,src)

    def test_narrow_single_pixel_box_quantization_is_quarantined(self):
        row,src=fixtures()
        src["start_bbox"]=[21,270,21,270]
        src["completion"]["start_point"]=[21,270]
        row["completion"]=json.dumps(src["completion"],sort_keys=True,separators=(",",":"))
        self.assertNotEqual(relative_to_source_pixel(
            source_pixel_to_relative([21,270],1024,640),1024,640),[21,270])
        with self.assertRaisesRegex(ValueError,"quantization escapes"):
            derive_two_independent_points(row,src)

    def test_no_label_leakage_from_metadata(self):
        row,src=fixtures()
        first=derive_two_independent_points(row,src)
        modified=deepcopy(src)
        modified["train_metadata_hidden_note"]="some unrelated metadata"
        self.assertEqual(derive_two_independent_points(row,modified),first)

    def test_endpoint_instruction_and_invalid_pixels(self):
        self.assertNotEqual(point_prompt("Drag the item",True,"start"),
                            point_prompt("Drag the item",True,"end"))
        with self.assertRaises(ValueError):point_prompt("Drag the item",True,"middle")
        with self.assertRaises(ValueError):source_pixel_to_relative([-1,10],1024,640)
        with self.assertRaises(ValueError):relative_to_source_pixel([1001,10],1024,640)

if __name__=="__main__":
    unittest.main()

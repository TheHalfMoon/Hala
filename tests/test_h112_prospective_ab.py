"""H112 native Qwen [0,1000] relative strict H61 source pixel test."""
import json,unittest
from h112.eval import frozen,START,END,COUNT,ARMS,inputs_for
from h112.coordinates import normalized_prompt,convert
from h61.official_eval import score

class TestH112Prospective(unittest.TestCase):
    def test_fresh_original_eval_56_through_59(self):
        self.assertEqual((START,COUNT,END),(56,4,60))
        self.assertEqual(ARMS,("base_pixels","base_normalized","h106_pixels","h106_normalized"))
        scenes=[{"id":str(i)} for i in range(60)]
        self.assertEqual([s["id"] for s in frozen(scenes)],["56","57","58","59"])
        with self.assertRaises(ValueError):frozen(scenes[:-1])
    def test_native_normalized_to_original_strict_hit_with_unchanged_scorer(self):
        scene={"id":"synthetic-only","domain":"sheet","image_width":1024,"image_height":640,
               "prompt":"Unchanged original source user instruction",
               "image_sha256":"a"*64,
               "task":{"intent":"Drag the item from its source to target","ordered":True,
                       "start_bbox":[510,318,514,322],
                       "end_bbox":[766,478,770,482]}}
        raw=json.dumps({"type":"drag","start_point":[500,500],"end_point":[750,750],
                        "coordinate_space":"normalized_0_1000","ordered":True})
        native=convert(raw,1024,640)
        self.assertIsNotNone(native)
        self.assertEqual(score([scene],{"synthetic-only":native})["task_success"],1)
        pixel=inputs_for([scene],False)[0]
        relative=inputs_for([scene],True)[0]
        self.assertEqual(pixel["image_sha256"],relative["image_sha256"])
        self.assertEqual(pixel["task"],relative["task"])
        self.assertEqual(pixel["prompt"],scene["prompt"])
        self.assertNotIn("start_bbox",relative["prompt"])
        self.assertIn("normalized",relative["prompt"].lower())
        self.assertNotEqual(pixel["prompt"],relative["prompt"])
if __name__=="__main__":unittest.main()

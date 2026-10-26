"""Source-only H129 marker feedback geometry/paired score regression."""
import unittest
from PIL import Image,ImageChops
from h129.feedback import draw_marker,feedback_prompt,score,validate_point
from h129.source import TOTAL,PREVIOUS,PER_FAMILY,FAMILIES,H128_SHA,propose

def fixture():
    cases=[];responses=[]
    for i in range(16):
        identity=f"synthetic-{i}"
        sha=f"sha-{i}"
        cases.append({"id":identity,"image_sha256":sha,"family":FAMILIES[i//4],
                      "image_width":800,"image_height":600,
                      "start_bbox":[100,100,106,106],
                      "end_bbox":[600,400,606,406]})
        responses.append({"id":identity,"image_sha256":sha,
                          "direct_start":(102,102),"direct_end":(602,402),
                          "feedback_start":(102,102),"feedback_end":(602,402)})
    return cases,responses

class H129Tests(unittest.TestCase):
    def test_source_disjoint_and_prior_sha_pins(self):
        self.assertEqual(PREVIOUS,2704)
        self.assertEqual(TOTAL,16)
        self.assertEqual(PER_FAMILY,4)
        self.assertEqual(H128_SHA,"bd0f0099cc42a15d1fb11691964adbb2bd01ae13fbe2f039d2c8d714d3f30c2b")
        with self.assertRaises(ValueError):propose([],None)

    def test_marker_is_model_only_derived_rgb_copy(self):
        image=Image.new("RGB",(800,600),(25,30,35))
        original=image.copy()
        marked=draw_marker(image,(103,107))
        self.assertEqual(marked.size,image.size)
        self.assertIsNone(ImageChops.difference(original,image).getbbox())
        self.assertIsNotNone(ImageChops.difference(original,marked).getbbox())
        self.assertEqual(marked.getpixel((103,100)),(255,0,210))
        for point in [(-1,5),(0,float("nan")),(800,2)]:
            with self.assertRaises(ValueError):draw_marker(image,point)

    def test_role_prompts_avoid_private_gold(self):
        message=feedback_prompt("Move the sheet border toward the left","start")
        self.assertIn("previous model GUESS",message)
        self.assertIn("START",message)
        self.assertNotIn("start_bbox",message)
        with self.assertRaises(ValueError):feedback_prompt("drag","unknown")

    def test_strict_pairs_and_fail_closed(self):
        cases,preds=fixture()
        result=score(cases,preds)
        self.assertEqual(result["aggregate"]["direct_full"],16)
        self.assertEqual(result["aggregate"]["feedback_full"],16)
        self.assertFalse(result["is_real_interactive_environment"])
        preds[0]["feedback_start"]=(400,300)
        result=score(cases,preds)
        self.assertEqual(result["aggregate"]["feedback_full"],15)
        self.assertEqual(result["aggregate"]["changed_start"],1)
        preds[0]["id"]="incorrect"
        with self.assertRaises(ValueError):score(cases,preds)

if __name__=="__main__":
    unittest.main()

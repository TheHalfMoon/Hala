"""H112 fail-closed relative to original pixel mapping."""
import json,unittest
from h112.coordinates import normalized_prompt,convert,NATIVE_SPACE
from h61.official_eval import parse_action

def raw(a,b):
    return json.dumps({"type":"drag","start_point":a,"end_point":b,
                       "coordinate_space":NATIVE_SPACE,"ordered":True})

class TestH112Relative(unittest.TestCase):
    def test_original_frame_shapes_and_boundaries(self):
        for w,h in ((1024,640),(1024,576),(495,640),(828,640),(452,640),(853,640),(854,640)):
            action=parse_action(convert(raw([0,0],[1000,1000]),w,h),w,h)
            self.assertEqual(action["start_point"],[0,0])
            self.assertEqual(action["end_point"],[w-1,h-1])
            middle=parse_action(convert(raw([500,500],[250,750]),w,h),w,h)
            self.assertLessEqual(abs(middle["start_point"][0]-w/2),1)
            self.assertLessEqual(abs(middle["start_point"][1]-h/2),1)
    def test_reject_malformed_bounds_noninteger_duplicate_keys(self):
        invalid=["not JSON","{}",raw([-1,500],[200,200]),
                 raw([1001,500],[500,500]),raw([True,10],[500,500]),
                 raw([2.0,2],[500,500]),
                 raw([10,10],[10,10]).replace('"normalized_0_1000"','"source_image_pixels"'),
                 '{"type":"drag","type":"drag"}']
        for item in invalid:
            with self.subTest(item=item[:30]):
                self.assertIsNone(convert(item,1024,640))
    def test_original_box_labels_never_in_prompt(self):
        scene={"image_width":1024,"image_height":576,
               "task":{"intent":"Drag the slider toward the requested value.",
                       "ordered":False,"start_bbox":[1,2,3,4]}}
        p=normalized_prompt(scene)
        self.assertIn(NATIVE_SPACE,p)
        self.assertIn("ordered=false",p)
        self.assertNotIn("start_bbox",p)
        scene["task"]["start_bbox"]=[400,400,700,600]
        self.assertEqual(normalized_prompt(scene),p)
if __name__=="__main__":unittest.main()

"""H152 source-only new original heading edge-error controller."""
import unittest
from collections import Counter
from h152.horizontal_visual_search_ab import (
    OFFSET,N,X_OFFSETS,seed_scene,offcenter_visible_goal
)
class H152NewOriginalLateralSearchTests(unittest.TestCase):
    def test_fresh_original_bilingual_goal_and_four_layouts(self):
        self.assertEqual(OFFSET,184000)
        self.assertEqual(N,16)
        items=[seed_scene(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in items),
                         {("ar","doing"):4,("en","doing"):4,
                          ("ar","done"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in items}),4)
        self.assertEqual(X_OFFSETS,(0,-32,32,-64,64))
        with self.assertRaises(ValueError):
            seed_scene(164000)
    def test_offcenter_heading_stays_inside_native_visible_header(self):
        heading={"x":410,"y":170,"width":120,"height":32}
        point=offcenter_visible_goal(heading,184000)
        self.assertTrue(heading["x"]<point[0]<heading["x"]+heading["width"])
        self.assertTrue(heading["y"]<point[1]<heading["y"]+heading["height"])
        self.assertGreater(abs(point[0]-(heading["x"]+heading["width"]/2)),35)
        self.assertEqual(point,offcenter_visible_goal(heading,184000))
        with self.assertRaises(ValueError):
            offcenter_visible_goal({},184000)
if __name__=="__main__":
    unittest.main()

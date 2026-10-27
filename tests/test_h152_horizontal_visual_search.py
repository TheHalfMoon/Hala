"""H152 fresh owner-only visible horizontal point search contracts."""
import unittest
from collections import Counter
from h152.horizontal_visual_search import (
    OFFSET,N,LANES,HORIZONTAL_PERTURB,new_source,displaced_visible_heading
)
class H152VisibleHorizontalSearch(unittest.TestCase):
    def test_fresh_bilingual_tasks_and_bounded_horizontal_lanes(self):
        self.assertEqual(OFFSET,174000)
        self.assertEqual(N,16)
        self.assertEqual(LANES,(0,-40,40,-80,80))
        self.assertEqual(HORIZONTAL_PERTURB,38)
        scenes=[new_source(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in scenes),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        self.assertEqual(len({x["layout"] for x in scenes}),4)
        with self.assertRaises(ValueError):new_source(164000)
    def test_horizontal_prediction_noise_inside_original_visible_header(self):
        bbox={"x":420,"y":170,"width":120,"height":31}
        for attempt in (174000,174001):
            point=displaced_visible_heading(bbox,attempt)
            self.assertEqual(point[1],185.5)
            self.assertIn(point[0],(442,518))
            self.assertTrue(bbox["x"]<=point[0]<=bbox["x"]+bbox["width"])
            self.assertTrue(bbox["y"]<=point[1]<=bbox["y"]+bbox["height"])
        with self.assertRaises(ValueError):displaced_visible_heading({},174000)
if __name__=="__main__":
    unittest.main()

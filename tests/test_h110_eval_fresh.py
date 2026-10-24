"""H110 frozen source-disjoint pilot and strict reference mapping tests."""
import unittest
from h110.eval import frozen,center_in_roi,START,COUNT,END,ARMS
from h109.crop_geometry import CropFrame

class TestH110Pilot(unittest.TestCase):
    def test_new_source_eval_44_to_47_only(self):
        rows=[{"id":str(i)} for i in range(48)]
        selected=frozen(rows)
        self.assertEqual((START,END,COUNT),(44,48,4))
        self.assertEqual([x["id"] for x in selected],["44","45","46","47"])
        self.assertEqual(ARMS,("h106_original","h106_two_crops_512"))
        with self.assertRaises(ValueError):frozen(rows[:-1])
    def test_bbox_center_roi_posthoc_only(self):
        frame=CropFrame(1024,640,200,0,512,512)
        self.assertTrue(center_in_roi([250,250,280,280],frame))
        self.assertFalse(center_in_roi([800,250,830,280],frame))
if __name__=="__main__":unittest.main()

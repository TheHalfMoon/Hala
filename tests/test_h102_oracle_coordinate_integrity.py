"""H102 pure CPU oracle checks; no GPU/model nor benchmark labels in tests."""
import json
import unittest
from h61.official_eval import score
from h102.oracle import in_bounds,center,make_action,assert_oracle,invalid_bbox_reasons

class H102OracleContracts(unittest.TestCase):
    def setUp(self):
        self.s={"id":"synthetic","domain":"slider","image_width":1024,
                "image_height":640,
                "task":{"ordered":True,"start_bbox":[100,100,120,120],
                        "end_bbox":[600,400,640,440]}}
    def test_correct_coordinate_pair_passes_original_scorer(self):
        self.assertEqual(center(self.s["task"]["start_bbox"]),[110,110])
        self.assertTrue(assert_oracle(self.s,[110,110],[620,420]))
    def test_wrong_action_fails_original_scorer(self):
        raw=json.dumps(make_action([900,100],[900,150],True))
        self.assertEqual(score([self.s],{"synthetic":raw})["task_success"],0)
    def test_invalid_boxes_rejected(self):
        for box in ([0,0,1024,50],[0,0,50,640],[-1,0,4,4],
                    [10,10,1,1],[0,0,1,1.5],[False,0,1,1]):
            self.assertFalse(in_bounds(box,1024,640))
        self.s["task"]["start_bbox"]=[0,0,1024,10]
        with self.assertRaises(ValueError):assert_oracle(self.s,[10,5],[620,420])
    def test_ordered_reverse_rejected(self):
        raw=json.dumps(make_action([620,420],[110,110],True))
        self.assertEqual(score([self.s],{"synthetic":raw})["task_success"],0)
    def test_unordered_reverse_accepted(self):
        self.s["task"]["ordered"]=False
        raw=json.dumps(make_action([620,420],[110,110],False))
        self.assertEqual(score([self.s],{"synthetic":raw})["task_success"],1)
    def test_boolean_coordinates_not_int(self):
        self.assertFalse(in_bounds([False,0,4,4],1024,640))
        with self.assertRaises(ValueError):
            assert_oracle(self.s,[True,110],[620,420])

    def test_boundary_anomaly_classifies_half_open_without_silent_fix(self):
        self.assertEqual(invalid_bbox_reasons([0,0,1024,20],1024,640),
                         ("right_equals_width_half_open_candidate",))
        self.assertEqual(invalid_bbox_reasons([0,0,1025,20],1024,640),
                         ("right_exceeds_width",))
        self.assertEqual(invalid_bbox_reasons([0,0,10,640],1024,640),
                         ("bottom_equals_height_half_open_candidate",))
        self.assertEqual(invalid_bbox_reasons([0,0,10,639],1024,640),())

if __name__=="__main__": unittest.main()

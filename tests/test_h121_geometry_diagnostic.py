"""Aggregate-only H121 original TRAIN and model output contracts."""
import json
import unittest
from h121.geometry import (_distribution,strict_role,strict_xy,
                           summarize_model_predictions,summarize_model_absolute)

def point(role,x,y):
    return json.dumps({"role":role,"point":[x,y],
                       "coordinate_space":"normalized_0_1000"})
def delta(x,y):
    return json.dumps({"role":"end_delta","delta":[x,y],
                       "coordinate_space":"normalized_displacement_1000"})

class H121GeometryTests(unittest.TestCase):
    def test_neural_role_json_is_exact_and_bounded(self):
        self.assertEqual(strict_role(point("start",20,1000),"start"),[20,1000])
        self.assertEqual(strict_role(delta(-1000,1000),"end_delta"),[-1000,1000])
        self.assertIsNone(strict_role(point("start",20,1000),"end"))
        self.assertIsNone(strict_role('{"role":"start","point":[0,0],"point":[1,1],"coordinate_space":"normalized_0_1000"}',"start"))
        self.assertIsNone(strict_role(point("start",1001,0),"start"))
        self.assertIsNone(strict_role(delta(1200,0),"end_delta"))

    def test_prediction_only_delta_geometry_aggregates_no_labels(self):
        a=[point("start",200,100),point("start",990,900),"{invalid"]
        b=[delta(500,-400),delta(100,100),delta(100,0)]
        result=summarize_model_predictions(a,b)
        self.assertEqual(result["model_calls_first_start"],3)
        self.assertEqual(result["start_schema_valid"],2)
        self.assertEqual(result["end_delta_schema_valid"],3)
        self.assertEqual(result["paired_parsable"],2)
        self.assertEqual(result["predicted_end_off_original_image"],2)
        self.assertEqual(result["predicted_signed_displacement"]["count"],2)
        self.assertNotIn("image_id",json.dumps(result))
        self.assertTrue(result["no_true_EVAL_locations_or_boxes_used"])

    def test_absolute_geometry_makes_sensible_vectors(self):
        result=summarize_model_absolute(
            [point("start",100,200),point("start",1,2)],
            [point("end",600,200),point("end",1,2)])
        self.assertEqual(result["paired_parsable"],2)
        self.assertEqual(result["predicted_displacement"]["near_zero_fraction"],.5)
        self.assertEqual(result["predicted_displacement"]["count"],2)

    def test_no_raw_points_in_aggregate(self):
        result=_distribution([[500,-20],[-100,80]])
        self.assertEqual(result["count"],2)
        self.assertIsInstance(result["direction_buckets"],dict)
        self.assertNotIn("vectors",result)
        with self.assertRaises(ValueError):_distribution([[1001,0]])

if __name__=="__main__":unittest.main()

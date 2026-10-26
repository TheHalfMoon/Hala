"""Owner-only after-inference geometry scoring regression, synthetic only."""
import unittest
from h131.owner_diagnostics import distance_to_box

class OwnerDiagnosticTests(unittest.TestCase):
    def test_point_box_distance_and_strict_refusal(self):
        box={"x":100,"y":200,"width":40,"height":20}
        self.assertEqual(distance_to_box((105,205),box),0)
        self.assertEqual(distance_to_box((100,200),box),0)
        self.assertEqual(distance_to_box((90,205),box),10)
        self.assertEqual(distance_to_box((90,190),box),2**0.5*10)
        with self.assertRaises(ValueError):
            distance_to_box((float("nan"),205),box)
        with self.assertRaises(ValueError):
            distance_to_box((1,2),dict(box,width=0))
if __name__=="__main__":unittest.main()

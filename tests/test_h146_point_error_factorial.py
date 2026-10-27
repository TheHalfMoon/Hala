"""H146 new owner-only four-way native point-error causal ablation."""
import unittest
from collections import Counter
from h146.point_error_factorial import (
    source,choose_points,ARMS,OFFSET,N
)
class H146OriginalPointErrorFactorialTests(unittest.TestCase):
    def test_new_task_ids_and_all_binary_noise_arms(self):
        self.assertEqual(N,16)
        self.assertEqual(OFFSET,102000)
        self.assertEqual(len(ARMS),4)
        scenes=[source(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((r["language"],r["destination_status"])
                                 for r in scenes),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        with self.assertRaises(ValueError):source(92000)
    def test_source_vs_goal_noise_are_separable_and_reproducible(self):
        a={"x":90,"y":250,"width":250,"height":100}
        b={"x":410,"y":160,"width":150,"height":30}
        clean=choose_points(a,b,102000,"clean")
        src=choose_points(a,b,102000,"source_noise")
        goal=choose_points(a,b,102000,"heading_noise")
        both=choose_points(a,b,102000,"both_noise")
        self.assertEqual(clean[1],src[1])
        self.assertEqual(clean[0],goal[0])
        self.assertNotEqual(clean[0],src[0])
        self.assertNotEqual(clean[1],goal[1])
        self.assertEqual(both,(src[0],goal[1]))
        self.assertEqual(both,choose_points(a,b,102000,"both_noise"))
        with self.assertRaises(ValueError):
            choose_points(a,b,102000,"other")
if __name__=="__main__":
    unittest.main()

"""H145 original source-only model-point noise, never H143 reuse or hidden DOM."""
import unittest
from collections import Counter
from h145.noisy_point_action_ab import (
    source,jittered_points,OFFSET,N,ARMS
)
class H145NoisyPointActionContracts(unittest.TestCase):
    def test_fresh_owner_original_task_ids_roles(self):
        self.assertEqual(OFFSET,92000)
        self.assertEqual(N,16)
        self.assertEqual(ARMS,("direct","staged_sweep"))
        rows=[source(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((r["language"],r["destination_status"])
                                 for r in rows),
                         {("ar","doing"):4,("en","doing"):4,
                          ("ar","done"):4,("en","done"):4})
        self.assertEqual(len({r["layout"] for r in rows}),4)
        with self.assertRaises(ValueError):source(82000)
    def test_noise_stays_inside_visible_source_and_header_only(self):
        source_box={"x":90,"y":250,"width":250,"height":104}
        heading={"x":420,"y":170,"width":120,"height":31}
        p,g=jittered_points(source_box,heading,92000)
        self.assertTrue(source_box["x"]<p[0]<source_box["x"]+source_box["width"])
        self.assertTrue(source_box["y"]<p[1]<source_box["y"]+source_box["height"])
        self.assertTrue(heading["x"]<g[0]<heading["x"]+heading["width"])
        self.assertTrue(heading["y"]<g[1]<heading["y"]+heading["height"])
        self.assertEqual((p,g),jittered_points(source_box,heading,92000))
        with self.assertRaises(ValueError):
            jittered_points({},heading,92000)
if __name__=="__main__":unittest.main()

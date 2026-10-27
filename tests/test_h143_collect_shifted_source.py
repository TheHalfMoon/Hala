"""H143 real shifted-layout source and no historical/BLIND leakage."""
import unittest
from collections import Counter
from h143.collect_shifted_source import (
    source_case,layout_name,style,OFFSET,TRANSFORMS,CASES
)
class H143ShiftedLayoutSourceTests(unittest.TestCase):
    def test_all_shifted_css_never_original_model_training_geometry(self):
        self.assertEqual(CASES,16)
        self.assertEqual(OFFSET,73000)
        self.assertGreaterEqual(len(TRANSFORMS),4)
        self.assertTrue(all(x!=0 or y!=0 for x,y in TRANSFORMS.values()))
        examples=[source_case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((x["language"],x["destination_status"])
                                 for x in examples),
                         {("ar","doing"):4,("ar","done"):4,
                          ("en","doing"):4,("en","done"):4})
        self.assertEqual(len({x["title"] for x in examples}),16)
        self.assertEqual(len(set(x["layout"] for x in examples)),4)
        for item in examples:
            css=style(item["layout"])
            self.assertIn(".task_column",css)
            self.assertIn("transform:translate(",css)
            self.assertNotIn(".hide_drop",css)
            self.assertNotIn("opacity",css)
    def test_prior_source_and_wrong_shift_fail_closed(self):
        with self.assertRaises(ValueError):
            source_case(62000)
        with self.assertRaises(ValueError):
            style("original_fixed_layout")
if __name__=="__main__":
    unittest.main()

"""H151 new original neural inference-only source contract."""
import unittest
from collections import Counter
from h151.collect_shifted_source import source_case,OFFSET,CASES,TRANSFORMS
class H151NewNeuralSourceContracts(unittest.TestCase):
    def test_new_model_task_sources_and_four_shifted_layouts(self):
        self.assertEqual(OFFSET,164000)
        self.assertEqual(CASES,16)
        scenes=[source_case(i) for i in range(OFFSET,OFFSET+16)]
        self.assertEqual(Counter((r["language"],r["destination_status"])
                                 for r in scenes),
                         {("ar","doing"):4,("en","doing"):4,
                          ("ar","done"):4,("en","done"):4})
        self.assertEqual(set(r["layout"] for r in scenes),set(TRANSFORMS))
        with self.assertRaises(ValueError):source_case(154000)
if __name__=="__main__":
    unittest.main()

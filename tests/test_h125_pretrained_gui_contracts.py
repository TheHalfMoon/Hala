"""H125 two specialized GUI model native point contracts, no GPU."""
import unittest
from h125.model_contracts import MODELS,parse_point,prompt_for
from h125.source import PER_FAMILY,TOTAL,OLD

class H125GrounderContracts(unittest.TestCase):
    def test_source_new64_excludes_all_previous_2560_original_images(self):
        self.assertEqual(PER_FAMILY,16)
        self.assertEqual(TOTAL,64)
        self.assertEqual(OLD,2560)

    def test_two_native_coordinate_spaces_not_interchangeable(self):
        self.assertEqual(parse_point("uground_2b","(500,250)",1000,600),
                         (500,150))
        self.assertEqual(parse_point("showui_2b","[0.5, 0.25]",1000,600),
                         (500,150))
        self.assertIsNone(parse_point("showui_2b","(500,250)",1000,600))
        self.assertIsNone(parse_point("uground_2b","[0.5, 0.25]",1000,600))
        self.assertEqual(MODELS["uground_2b"].license,"apache-2.0")
        self.assertEqual(MODELS["showui_2b"].license,"mit")

    def test_refuses_code_indirection_extra_keys_or_oob(self):
        for raw in ("__import__('os').system('echo bad')",
                    "[1,2,3]",'{"x":0.5,"y":0.5}',
                    "[-1,5]","[nan, 0.3]","[1.4,0.3]"):
            self.assertIsNone(parse_point("showui_2b",raw,1000,600))
        self.assertIsNone(parse_point("not_a_model","(10,20)",1000,600))
        self.assertIsNone(parse_point("uground_2b","(1001,20)",1000,600))
        self.assertIsNone(parse_point("uground_2b","(True,20)",1000,600))

    def test_no_oracle_coordinates_in_inference_prompt(self):
        task="Resize the presentation image by dragging the lower handle"
        for model in MODELS:
            for role in ("START","END"):
                p=prompt_for(model,task,role)
                self.assertIn(task,p)
                self.assertIn(role,p)
                self.assertNotIn("start_bbox",p)
                self.assertNotIn("end_bbox",p)
        with self.assertRaises(ValueError):prompt_for("uground_2b","x","START")
        with self.assertRaises(ValueError):prompt_for("uground_2b",task,"BOTH")

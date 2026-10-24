"""H101 deterministic CPU-only geometry-prompt intervention preregistration."""
import unittest
from h101 import eval as h101
from h61.official_eval import parse_action
class H101ContractTests(unittest.TestCase):
    def test_disjoint_original_upstream_public_eval_sample(self):
        self.assertEqual((h101.START_INDEX,h101.PER_DOMAIN,h101.FETCH_COUNT),(16,4,20))
        self.assertEqual(h101.select_unseen([{"id":str(i)} for i in range(20)]),
                         [{"id":str(i)} for i in range(16,20)])
        for old in (range(0,4),range(4,8),range(8,12),range(12,16)):
            self.assertTrue(set(range(16,20)).isdisjoint(old))
        with self.assertRaises(ValueError):
            h101.select_unseen([{"id":str(i)} for i in range(19)])
    def test_frozen_2x2_no_optimizer(self):
        self.assertEqual(h101.ARMS,("base_original","base_geometry_prompt",
                                    "h99_original","h99_geometry_prompt"))
        self.assertEqual(len(h101.DOMAINS),4)
    def test_intervention_uses_no_reference_answers_or_boxes(self):
        original={"id":"scene","prompt":"Move the icon. Return JSON.","image_width":1024,
                  "image_height":640,"task":{"start_bbox":[1,2,3,4],"end_bbox":[5,6,7,8]}}
        mutated={**original,"task":{"start_bbox":[700,700,800,800]}}
        a=h101.coordinate_prompt_from_scene(original)
        b=h101.coordinate_prompt_from_scene(mutated)
        self.assertEqual(a,b)
        self.assertTrue(a.startswith(original["prompt"]))
        self.assertIn("x=0..1023",a)
        self.assertIn("y=0..639",a)
        self.assertNotIn("700",a)
        self.assertNotIn("start_bbox",a)
        self.assertNotIn("end_bbox",a)
    def test_rejects_invalid_dimensions_and_duplicate_scenes(self):
        with self.assertRaises(ValueError):
            h101.coordinate_prompt_from_scene({"prompt":"a","image_width":0,"image_height":640})
        with self.assertRaises(ValueError):
            h101.select_unseen([{"id":str(i)} for i in range(19)]+[{"id":"16"}])
    def test_original_strict_json_scorer_not_weakend(self):
        literal='{"type":"drag","start_point":[10,20],"end_point":[30,40],"coordinate_space":"source_image_pixels","ordered":true}'
        self.assertIsNotNone(parse_action(literal,1024,640))
        self.assertIsNone(parse_action("```json\n"+literal+"\n```",1024,640))
    def test_private_kernel_and_adapter_required(self):
        self.assertEqual(len(h101.H99_ADAPTER_SHA),64)
        with self.assertRaises(ValueError):
            h101.require_owner_private_runtime(env={},system="Linux",work="/kaggle/tmp/a",output="/kaggle/working/a")
if __name__=="__main__":
    unittest.main()

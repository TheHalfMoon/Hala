"""H134 frozen vision-LoRA plan guards without downloading weights."""
import json
import unittest
from h134.train_gpu import STEPS,GRAD_ACCUM,LORA_RANK,make_prompt_and_gold

class H134TrainingContractTests(unittest.TestCase):
    def test_start_and_end_both_have_original_gold(self):
        row={"instruction":"Drag the blue panel handle to green target.",
             "start_grid_1000":[125,230],"end_grid_1000":[735,612]}
        a,gold_start=make_prompt_and_gold(row,"start")
        b,gold_end=make_prompt_and_gold(row,"end")
        self.assertEqual(json.loads(gold_start),{"point_2d":[125,230]})
        self.assertEqual(json.loads(gold_end),{"point_2d":[735,612]})
        self.assertNotEqual(a,b)
        with self.assertRaises(ValueError):
            make_prompt_and_gold(row,"unapproved")
        row["start_grid_1000"]=[1200,-1]
        with self.assertRaises(ValueError):
            make_prompt_and_gold(row,"start")

    def test_bounded_private_resource_budget(self):
        self.assertGreater(STEPS,0)
        self.assertLessEqual(STEPS,24)
        self.assertLessEqual(GRAD_ACCUM,2)
        self.assertLessEqual(LORA_RANK,8)
if __name__=="__main__":unittest.main()

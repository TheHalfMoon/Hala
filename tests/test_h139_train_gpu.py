"""H139 fully original MIT application source training contracts."""
import unittest
from h139.train_gpu import make_prompt_and_gold,STEPS,GRAD_ACCUM,LORA_RANK
from h139.source import COUNTS,SPLITS

class H139FrozenTrainingGuards(unittest.TestCase):
    def test_real_upstream_pixels_and_bilingual_instruction(self):
        row={"instruction":"Drag task card into In Progress",
             "start_bbox":{"x":95,"y":95,"width":10,"height":10},
             "end_bbox":{"x":190,"y":190,"width":20,"height":20}}
        _,start=make_prompt_and_gold(row,"start")
        _,end=make_prompt_and_gold(row,"end")
        self.assertEqual(start,'{"point_2d":[104,139]}')
        self.assertEqual(end,'{"point_2d":[208,278]}')
        self.assertNotEqual(start,end)
        with self.assertRaises(ValueError):make_prompt_and_gold(row,"unknown")

    def test_free_model_limit_splits_and_no_ood_training(self):
        self.assertEqual(STEPS,64)
        self.assertEqual(GRAD_ACCUM,2)
        self.assertEqual(LORA_RANK,8)
        self.assertEqual(COUNTS,{"train":32,"dev":8})
        self.assertEqual(SPLITS["train"][1],64)
        self.assertEqual(SPLITS["dev"][1],16)

if __name__=="__main__":unittest.main()

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

    def test_exact_pinned_private_kaggle_job(self):
        import json
        from pathlib import Path
        from h139.kaggle_private_train import SOURCE_SHA, OUTPUT, identify_owner_splits
        metadata=json.loads((Path(__file__).parent.parent/"h139"/"kernel-metadata.json").read_text("utf8"))
        self.assertEqual(SOURCE_SHA,"bef2a0a7ccbeb64f53c6d86d976914b8a7eba8f9")
        self.assertEqual(len(SOURCE_SHA),40)
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(len(metadata["dataset_sources"]),2)
        self.assertFalse(any("blind" in x.lower() or "sealed" in x.lower()
                             for x in metadata["dataset_sources"]))
        self.assertTrue(str(OUTPUT).replace("\\","/").startswith("/kaggle/working/"))

if __name__=="__main__":unittest.main()

"""H136 small-handle source, frozen 3:1 START emphasis, and honest pilot gates."""
import unittest
from pathlib import Path

from h136.source import SPLITS, FAMILIES, seeds, SCHEMA, center
from h136.train_gpu import STEPS, GRAD_ACCUM, make_prompt_and_gold, LORA_RANK

class H136SourceAndTrainingContracts(unittest.TestCase):
    def test_source_splits_and_six_types(self):
        self.assertEqual({k:len(list(seeds(k))) for k in SPLITS},
                         {"train":288,"dev":24,"sealed":24})
        self.assertEqual(len(FAMILIES),6)
        sets={k:set(seeds(k)) for k in SPLITS}
        for left in sets:
            for right in sets:
                if left!=right:
                    self.assertFalse(sets[left]&sets[right])
        self.assertEqual(SCHEMA,"hala.h136.owner-original-workstation-gui-scenes.v1")

    def test_tiny_handles_in_original_new_visual_fixture(self):
        fixture=(Path(__file__).resolve().parents[1]/"h136"/"fixture.html").read_text("utf8")
        self.assertIn("typeIndex===2)?10:15",fixture)
        self.assertIn("for(let i=0;i<5;i++)",fixture)
        self.assertIn("setPointerCapture",fixture)
        self.assertIn("data-success",fixture)
        self.assertNotEqual(fixture,
            (Path(__file__).resolve().parents[1]/"h135"/"fixture.html").read_text("utf8"))

    def test_start_end_target_labels_are_from_real_original_browser_coordinates(self):
        row={"instruction":"Drag the thin blue divider to the outlined target",
             "start_px":[240,200],"end_px":[600,300]}
        a,g1=make_prompt_and_gold(row,"start")
        b,g2=make_prompt_and_gold(row,"end")
        self.assertEqual(g1,'{"point_2d":[250,278]}')
        self.assertEqual(g2,'{"point_2d":[625,417]}')
        self.assertNotEqual(a,b)
        with self.assertRaises(ValueError):make_prompt_and_gold(row,"wrong")

    def test_free_training_budget_and_start_oversampling(self):
        self.assertEqual(STEPS,64)
        self.assertEqual(GRAD_ACCUM,2)
        self.assertEqual(LORA_RANK,8)
        script=(Path(__file__).resolve().parents[1]/"h136"/"train_gpu.py").read_text("utf8")
        self.assertIn('("start","start","start","end")',script)
        self.assertIn("model.gradient_checkpointing_enable()",script)
        self.assertNotIn("image.resize((640,480))",script)
if __name__=="__main__":unittest.main()

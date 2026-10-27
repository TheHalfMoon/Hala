"""H161 fail-closed private multimodal SFT source/no-GPU gold leakage."""
import json
import inspect
import unittest
from h161.train_gpu import (
    admit, actions, prompt, label, STEPS, ACCUM, SEED,
)


class H161MixedOriginalSFTSafetyTests(unittest.TestCase):
    def test_action_schema_rejects_bad_native_coords_and_extra_fields(self):
        accepted=(
            {"actions":[{"kind":"click","point":[128,224]}]},
            {"actions":[{"kind":"drag","start":[120,200],"end":[520,220]}]},
        )
        for doc in accepted:
            self.assertTrue(actions(json.dumps(doc))["valid"])
        rejected=(
            {"actions":[{"kind":"click","point":[960,1]}]},
            {"actions":[{"kind":"drag","start":[1,2],"end":[22,-1]}]},
            {"actions":[{"kind":"click","point":[12,13],"dom":"button"}]},
            {"actions":[{"kind":"drag","start":[1,2],"end":[3,4]},
                        {"kind":"click","point":[1,2]}]},
        )
        for doc in rejected:
            self.assertFalse(actions(json.dumps(doc))["valid"])

    def test_bilingual_native_prompt_and_train_label(self):
        base={"app":"react-dnd-kit-board","instruction":"اسحب البطاقة",
              "training_supervision":{"actions":[
                  {"kind":"drag","start":[50,60],"end":[400,180]}]}}
        self.assertIn("اسحب البطاقة",prompt(base))
        self.assertIn("drag",prompt(base))
        self.assertEqual(actions(label(base))["kind"],"drag")

    def test_fixed_train_only_two_epoch_plan(self):
        self.assertEqual((STEPS,ACCUM,STEPS*ACCUM,SEED),(48,2,96,161))
        import h161.train_gpu as m
        source=inspect.getsource(m.run)
        self.assertIn('admit(train_archive,"TRAIN")',source)
        self.assertIn('admit(dev_archive,"DEV")',source)
        self.assertIn('model(**batch)',source)
        self.assertIn('schedule=list(range(48))*2',source)
        self.assertIn('dev_labels_seen_by_GPU',source)
        self.assertNotIn('h140.private',source)


if __name__=="__main__":
    unittest.main()

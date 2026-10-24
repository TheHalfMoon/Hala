"""H99 regression: existing H61/H97 default, source-bound balanced study."""
import hashlib
import inspect
import json
import unittest
from pathlib import Path

from h99.research_train import training_config, STEPS, SCHEMA
from h97.research_train import training_config as old_vision_config
from h99.sampler import STRATEGY, build_balanced_plan
from training.scripts.train_adapter import train

class H99QualifiedStudy(unittest.TestCase):
    def test_128_original_coordinate_weighted_steps(self):
        repo=Path(__file__).resolve().parents[1]
        older=old_vision_config(repo)
        config=training_config(repo)
        self.assertEqual(STEPS,128)
        self.assertEqual(config["max_steps"],128)
        self.assertEqual({k:v for k,v in config.items() if k!="max_steps"},
                         {k:v for k,v in older.items() if k!="max_steps"})
        self.assertEqual(config["coordinate_weight"],8.0)
    def test_no_change_original_default_trainer(self):
        sig=inspect.signature(train)
        self.assertIsNone(sig.parameters["sampling_strategy"].default)
        self.assertFalse(sig.parameters["vision_lora"].default)
        self.assertFalse(sig.parameters["resume"].default)
        self.assertIsNone(sig.parameters["init_adapter"].default)
    def test_experiment_identity_is_separate(self):
        self.assertIn("h99",STRATEGY)
        self.assertIn("h99",SCHEMA)
    def test_private_plan_digest_is_not_source_id_list(self):
        from tests.test_h99_balanced_train_sampler import fake
        records=[fake(i) for i in range(256)]
        plan=build_balanced_plan(records,128)
        self.assertEqual(plan.summary["selected_languages"]["ar"]+
                         plan.summary["selected_languages"]["en"],128)
        self.assertEqual(plan.summary["unique_train_record_ids_selected"],128)
        self.assertEqual(len(plan.sha256),64)
        self.assertFalse(any(row["id"] in json.dumps(plan.summary) for row in records))
    def test_no_missing_language_in_eligible_pilot(self):
        from tests.test_h99_balanced_train_sampler import fake
        data=[fake(i,lang="ar" if i<40 else "en") for i in range(250)]
        plan=build_balanced_plan(data,64)
        self.assertGreater(plan.summary["selected_languages"]["ar"],0)
        self.assertGreater(plan.summary["selected_languages"]["en"],0)

if __name__=="__main__":
    unittest.main()

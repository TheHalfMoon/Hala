"""H107 frozen fresh source case selector, sha identity and fail-closed admission."""
import hashlib
import json
import unittest
from h107.eval import select_fresh,plan_sha,expected_training_identity
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h106.research_train import config
from training.scripts.train_adapter import JOINT_LORA_SCOPE_NONCE
from h106.training_plan import STRATEGY as H106_STRATEGY


class H107FrozenSource(unittest.TestCase):
    def setUp(self):
        self.samples=[{"id":f"eval-source-{i}","prompt":"original prompt",
                       "image_sha256":f"{i:064x}"} for i in range(36)]

    def test_original_public_eval_20_to_35_only(self):
        chosen=select_fresh(self.samples)
        self.assertEqual(len(chosen),16)
        self.assertEqual(chosen[0]["id"],"eval-source-20")
        self.assertEqual(chosen[-1]["id"],"eval-source-35")
        self.assertTrue(all(x["id"] not in {s["id"] for s in self.samples[:20]}
                            for x in chosen))

    def test_malformed_or_duplicate_eval_refused(self):
        with self.assertRaises(ValueError):select_fresh(self.samples[:35])
        scenes=list(self.samples);scenes[30]=scenes[20]
        with self.assertRaises(ValueError):select_fresh(scenes)

    def test_h106_source_identity_is_original_512_h105_prefix(self):
        self.assertEqual(len(H105_QUALIFIED_PLAN_SHA),64)
        self.assertRegex(plan_sha(),r"^[0-9a-f]{64}$")
        self.assertNotEqual(plan_sha(),H105_QUALIFIED_PLAN_SHA)

    def test_expected_train_identity_fails_if_training_config_changes(self):
        expected=expected_training_identity(".")
        self.assertRegex(expected,r"^[0-9a-f]{64}$")
        cfg=config(".")
        signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        signature=hashlib.sha256((signature+JOINT_LORA_SCOPE_NONCE).encode()).hexdigest()
        independent=hashlib.sha256((signature+H106_STRATEGY+plan_sha()).encode()).hexdigest()
        self.assertEqual(expected,independent)

if __name__=="__main__":unittest.main()

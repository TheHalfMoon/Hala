"""H124 original trained binaries are independent SHA locked before heldout."""
import unittest
from pathlib import Path
from h124 import admit_checkpoint,heldout_source
from h124.eval_heldout import SCHEMA as EVAL_SCHEMA

class H124RealHeldoutAdmissionTests(unittest.TestCase):
    def test_frozen_all_three_owner_verified_model_SHA(self):
        self.assertEqual(admit_checkpoint.ADAPTER_SHA,
             "509655e618defe8b5677e9235e4a281da63fbb0914c68738231b0a64956f122d")
        self.assertEqual(admit_checkpoint.CHECKPOINT_SHA,
             "94397d8869bb49439ac08ca40cde919190d3c58563fd8542c96a0849cd1b9e84")
        self.assertEqual(admit_checkpoint.OPTIMIZER_SHA,
             "a9080a58831ad7b7e1759fb54be5118d18b0b57b59e0e78401772ce935e03ac9")
        with self.assertRaisesRegex(ValueError,"private model kernel attachment absent"):
            admit_checkpoint.admit(Path("no-model"),Path("."),None)

    def test_only_original_train128_source_accepted(self):
        with self.assertRaises(ValueError):
            heldout_source.admit_original_train128([],Path("."),None)
        self.assertIn("original-TRAIN128",EVAL_SCHEMA)

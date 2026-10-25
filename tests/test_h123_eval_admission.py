"""H123 future original heldout must be unarmed absent actual verified weights."""
import unittest
from pathlib import Path
from h123 import admit_checkpoint,heldout_source
from h123.eval_heldout import SCHEMA as EVAL_SCHEMA

class H123PrivateEvalAdmissionTests(unittest.TestCase):
    def test_real_training_model_sha_pins_unarmed_until_owner_replay(self):
        self.assertIsNone(admit_checkpoint.ADAPTER_SHA)
        self.assertIsNone(admit_checkpoint.CHECKPOINT_SHA)
        self.assertIsNone(admit_checkpoint.OPTIMIZER_SHA)
        with self.assertRaisesRegex(ValueError,"not yet admitted"):
            admit_checkpoint.admit(Path("no-source"),Path("no-repo"),None)

    def test_missing_or_forged_original_train_source_unqualified(self):
        with self.assertRaises(ValueError):
            heldout_source.admit_original_train128([],Path("."),None)
        self.assertIn("real-private-original-TRAIN128",EVAL_SCHEMA)

if __name__=="__main__":
    unittest.main()

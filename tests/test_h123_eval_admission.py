"""H123 future original heldout must be unarmed absent actual verified weights."""
import unittest
from pathlib import Path
from h123 import admit_checkpoint,heldout_source
from h123.eval_heldout import SCHEMA as EVAL_SCHEMA

class H123PrivateEvalAdmissionTests(unittest.TestCase):
    def test_owner_verified_real_training_byte_sha_pins(self):
        self.assertEqual(admit_checkpoint.ADAPTER_SHA,
            "c8fe571d742d9d53633d5565e2a7edc5d78e42bd655c2d920aa9750096d5e5ff")
        self.assertEqual(admit_checkpoint.CHECKPOINT_SHA,
            "78d098528078b7a825a36b6b9af0978e47e481cada3ce374afae82a46637df47")
        self.assertEqual(admit_checkpoint.OPTIMIZER_SHA,
            "79377ab1057e2e60331dc1d11ea1648684567fd01173edc3dca6792a4bd414a7")
        with self.assertRaisesRegex(ValueError,"private Kaggle checkpoint attachment"):
            admit_checkpoint.admit(Path("no-source"),Path("no-repo"),None)

    def test_missing_or_forged_original_train_source_unqualified(self):
        with self.assertRaises(ValueError):
            heldout_source.admit_original_train128([],Path("."),None)
        self.assertIn("real-private-original-TRAIN128",EVAL_SCHEMA)

if __name__=="__main__":
    unittest.main()

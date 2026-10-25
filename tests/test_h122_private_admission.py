"""H122 heldout GPU admits only exact independently verified real model bytes.

CPU tests require no Kaggle quota, weights or local inference.
"""
import unittest
from pathlib import Path

from h122.admit_checkpoint import (
    H122_VERIFIED_ADAPTER_SHA,
    H122_VERIFIED_CHECKPOINT_SHA,
    H122_VERIFIED_OPTIMIZER_SHA,
    admit_checkpoint)
from h122.heldout_source import admit_original_train_heldout
from h122.schedule import Schedule


class H122PrivateResearchGateTest(unittest.TestCase):
    def test_owner_verified_real_training_hashes_are_pinned(self):
        self.assertEqual(H122_VERIFIED_ADAPTER_SHA,
            "c870d3c37ffb4d4811ddd60539a697f4867e20f65c2e9b4d436083a807326efc")
        self.assertEqual(H122_VERIFIED_CHECKPOINT_SHA,
            "2659b4cf2e256c1383114cadd493c89b1eba205aca5e149f9cf4201f0aee0d8b")
        self.assertEqual(H122_VERIFIED_OPTIMIZER_SHA,
            "b83675126a3499fe448ce3b05a79de2b59fef3193a69636de637e17d27267964")
        # Even SHA-pinned weights cannot bypass private checkpoint and source.
        with self.assertRaisesRegex(ValueError,"private attached Kaggle"):
            admit_checkpoint(Path("missing"),Path("missing"),[],
                             Path("missing"),None)

    def test_heldout_source_needs_authentic_training_split_before_media(self):
        with self.assertRaises(ValueError):
            admit_original_train_heldout([],Path("missing"),None)
        bogus=Schedule(train=(),heldout=(),train_original_indices=(),
                       heldout_original_indices=(),
                       summary={"source_split":"TRAIN_ONLY",
                                "no_holdout_optimizer_input":True,
                                "no_EVAL_or_TEST_source":True})
        with self.assertRaises(ValueError):
            admit_original_train_heldout([],Path("missing"),bogus)

if __name__=="__main__":
    unittest.main()

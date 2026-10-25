"""H122 checkpoint and heldout GPU runners are UNARMED until owner SHA proof.

These tests require no Kaggle quota, no model weights or local inference.
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
    def test_unverified_training_binary_cannot_be_evaluated(self):
        self.assertIsNone(H122_VERIFIED_ADAPTER_SHA)
        self.assertIsNone(H122_VERIFIED_CHECKPOINT_SHA)
        self.assertIsNone(H122_VERIFIED_OPTIMIZER_SHA)
        with self.assertRaisesRegex(ValueError,"not yet admitted"):
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

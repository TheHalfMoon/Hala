"""H124 optimizer start/END geometry gating; no GPU required in CI."""
import unittest
from h124.objective import completion
from h124.supervision import spatial_weights
from h124.training_plan import (TRAIN_SHA,HELDOUT_SHA,
                                 SCHEDULE_SHA,STEPS,checkpoint_due,get_plan)
from training.scripts.train_adapter import (train,H124_JOINT_SCOPE_NONCE)
from h106.training_plan import STRATEGY


class H124TrainSafetyTests(unittest.TestCase):
    def test_verified_original_896_new_split_SHA(self):
        self.assertEqual(STEPS,896)
        self.assertEqual(len({TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA}),3)
        for s in (TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA):
            self.assertEqual(len(s),64)
            self.assertTrue(all(letter in "0123456789abcdef" for letter in s))
        self.assertIn("h124",H124_JOINT_SCOPE_NONCE)
        self.assertTrue(checkpoint_due(896))
        self.assertTrue(checkpoint_due(64))
        self.assertFalse(checkpoint_due(63))
        with self.assertRaises(ValueError):checkpoint_due(897)
        with self.assertRaises(ValueError):get_plan(None)

    def test_numeric_losses_include_negative_joint_delta_digits(self):
        text=completion([700,650],[300,100])
        ids=list(range(len(text)))
        weights=spatial_weights(ids,lambda seq:text[seq[0]],8.0)
        self.assertEqual(len(ids),len(weights))
        self.assertTrue(any(w>1 for w in weights))
        self.assertIn("-400",text)
        self.assertIn("-550",text)
        delta_index=text.index("-400")
        self.assertEqual(weights[delta_index],8.0)
        self.assertTrue(all(w in (1.0,8.0) for w in weights))
        with self.assertRaises(ValueError):
            spatial_weights([0],lambda seq:"x",8.0)

    def test_prior_adapters_and_other_modes_refused(self):
        base={"joint_lora":True,"sampling_strategy":STRATEGY}
        for additional in (
            {"joint_lora":False},
            {"complex_grid_mode":True},
            {"resume":True},
            {"vision_lora":True},
            {"init_adapter":"h123"},
            {"point_sft_path":"old_h114"},
            {"end_grid_sft_path":"old_h122"},
            {"stop_after_step":32},
            {"sampling_strategy":None}
        ):
            with self.subTest(additional=additional):
                args={**base,**additional}
                with self.assertRaises(ValueError):
                    train("missing","missing","missing","missing","missing",
                          joint_geometry_mode=True,**args)


if __name__=="__main__":
    unittest.main()

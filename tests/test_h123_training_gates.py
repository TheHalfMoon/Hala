"""H123 real isolated optimizer path and spatial targets; no GPU needed."""
import json
import unittest
from h123.objective import completion
from h123.supervision import spatial_weights
from h123.training_plan import (STEPS,TRAIN_SHA,HELDOUT_SHA,
                                 SCHEDULE_SHA,get_plan,checkpoint_due)
from h123.schedule import Schedule
from training.scripts.train_adapter import train,H123_COMPLEX_SCOPE_NONCE

class H123TrainingGateTests(unittest.TestCase):
    def test_real_original_source_frozen_research_ids(self):
        self.assertEqual(STEPS,896)
        for value in (TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA):
            self.assertEqual(len(value),64)
            self.assertTrue(all(x in '0123456789abcdef' for x in value))
        self.assertIn("h123",H123_COMPLEX_SCOPE_NONCE)
        self.assertTrue(checkpoint_due(896))
        self.assertTrue(checkpoint_due(64))
        self.assertFalse(checkpoint_due(63))
        with self.assertRaises(ValueError): checkpoint_due(897)

    def test_weight_original_quadrant_cell_point_only(self):
        target=completion([435,561])
        ids=list(range(len(target)))
        weights=spatial_weights(ids,lambda i:target[i[0]],8.0)
        self.assertEqual(len(weights),len(target))
        self.assertTrue(all(w in (1.0,8.0) for w in weights))
        self.assertTrue(any(w>1 for w in weights))
        marker=target.index("normalized_0_1000")
        self.assertTrue(all(x==1 for x in weights[marker:marker+19]))
        with self.assertRaises(ValueError):
            spatial_weights(list(range(2)),lambda i:"?",8.0)

    def test_fake_original_train_source_cannot_satisfy_h123_plan(self):
        with self.assertRaises(ValueError):get_plan(None)
        wrong=Schedule(train=tuple({"original_source_id":str(i)} for i in range(896)),
                       heldout=tuple({"original_source_id":str(i)} for i in range(896,1024)),
                       train_indices=tuple(range(896)),
                       heldout_indices=tuple(range(896,1024)),
                       summary={"source_split":"TRAIN_ONLY","train_private_sha256":"0"*64})
        with self.assertRaises(ValueError):get_plan(wrong)

    def test_optin_only_fresh_gpu_mode(self):
        base={"joint_lora":True,
              "sampling_strategy":"hala.h106.joint-vision-language-h105-source-prefix256.v1"}
        for override in ({"joint_lora":False},{"sampling_strategy":None},
                         {"resume":True},{"vision_lora":True},
                         {"end_grid_sft_path":"old_h122"},
                         {"init_adapter":"old_model"},
                         {"point_sft_path":"prior"},
                         {"complex_grid_mode":True,"stop_after_step":32}):
            args={**base,**override}
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    train("missing","missing","missing","missing","missing",
                          complex_grid_mode=True,**{k:v for k,v in args.items()
                            if k!="complex_grid_mode"})


if __name__=="__main__":
    unittest.main()

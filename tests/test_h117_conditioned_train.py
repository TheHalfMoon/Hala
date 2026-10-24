"""H117 no-leakage opt-in point+signed-delta training qualification."""
import json
import unittest
from h114.schedule import trainer_row
from h116.delta import derive_schedule,SPACE,DELTA_ROLE
from training.scripts.train_adapter import train,H117_DELTA_SCOPE_NONCE
from h61.supervision import coordinate_token_flags
from h117.supervision import target_only_weights,role_from_exact_completion,role_training_row

class TestH117ConditionedTraining(unittest.TestCase):
    def test_exact_signed_and_start_schemas_and_nonce(self):
        self.assertIn("h117.",H117_DELTA_SCOPE_NONCE)
        self.assertNotEqual(SPACE,"normalized_0_1000")
        self.assertEqual(DELTA_ROLE,"end_delta")
        self.assertIn("signed",H117_DELTA_SCOPE_NONCE)
    def test_optin_never_reinterprets_existing_h114_or_h106(self):
        with self.assertRaises(ValueError):
            train(None,None,None,None,None,conditioned_delta_sft_path="private",
                  joint_lora=False)
        with self.assertRaises(ValueError):
            train(None,None,None,None,None,conditioned_delta_sft_path="private",
                  point_sft_path="other",joint_lora=True)
    def test_overlay_keeps_real_source_provenance(self):
        source={"id":"source-train","split":"train","image_path":"source.png",
                "image_sha256":"a"*64,"language":"en","family":"slider",
                "image_format":"JPEG","prompt":"original prompt",
                "completion":"old"}
        end={"id":"source-train::end_delta","original_source_id":"source-train",
             "split":"train","endpoint_role":"end_delta",
             "image_path":"source.png","image_sha256":"a"*64,
             "language":"en","family":"slider",
             "prompt":"Source intent and bounded signed displacement instructions",
             "completion":json.dumps({"role":"end_delta","delta":[-150,220],
                            "coordinate_space":SPACE})}
        unchanged=dict(source)
        updated=trainer_row(source,end)
        self.assertEqual(source,unchanged)
        self.assertEqual(updated["image_format"],"JPEG")
        self.assertEqual(updated["completion"],end["completion"])
        self.assertEqual(updated["image_sha256"],source["image_sha256"])
        self.assertNotEqual(updated["prompt"],source["prompt"])
        repaired=role_training_row(source,end)
        self.assertEqual(repaired["endpoint_role"],"end_delta")
        self.assertEqual(repaired["image_sha256"],source["image_sha256"])
        self.assertNotIn("endpoint_role",source)
        with self.assertRaises(ValueError):
            role_training_row(source,dict(end,endpoint_role="end"))
    def test_negative_displacement_digits_remain_supervised(self):
        completion=json.dumps({"role":"end_delta","delta":[-221,55],
                               "coordinate_space":SPACE})
        chunks=[ord(c) for c in completion]
        flags=coordinate_token_flags(chunks,lambda ts:chr(ts[0]))
        self.assertGreater(sum(flags),5)  # old bug weights schema 1000 too
        weights=target_only_weights(chunks,lambda ts:chr(ts[0]),"end_delta")
        self.assertEqual(sum(weight>1 for weight in weights),6)
        self.assertEqual(weights[completion.index("-")],8.0)
        self.assertEqual(weights[completion.index("1000")],1.0)
    def test_start_target_metadata_0_1000_does_not_steal_coordinate_weight(self):
        completion=json.dumps({"role":"start","point":[123,456],
                               "coordinate_space":"normalized_0_1000"})
        weights=target_only_weights([ord(c) for c in completion],
                                    lambda ts:chr(ts[0]),"start")
        self.assertEqual(sum(w>1 for w in weights),6)
        self.assertEqual(weights[completion.index("1000")],1.0)
        with self.assertRaises(ValueError):
            target_only_weights([ord(c) for c in completion],
                                lambda ts:chr(ts[0]),"end_delta")
    def test_kaggle_regression_missing_endpoint_role_metadata(self):
        # Previous private T4 failed with KeyError('endpoint_role') before
        # training step one: original media overlay stores role in completion.
        source={"id":"source-train","split":"train","image_path":"source.png",
                "image_sha256":"a"*64,"language":"en","family":"slider"}
        for role,target in (
            ("start",{"role":"start","point":[230,750],
                      "coordinate_space":"normalized_0_1000"}),
            ("end_delta",{"role":"end_delta","delta":[-200,75],
                          "coordinate_space":"normalized_displacement_1000"})):
            derived={**source,"id":"source-train::"+role,
                     "original_source_id":"source-train",
                     "prompt":"Original source-bound role instruction",
                     "completion":json.dumps(target)}
            # No metadata role in model training row; JSON is authoritative.
            model_row=trainer_row(source,derived)
            self.assertNotIn("endpoint_role",model_row)
            self.assertEqual(role_from_exact_completion(model_row["completion"]),role)
        with self.assertRaises(ValueError):
            role_from_exact_completion(json.dumps({"role":"end_delta",
                "delta":[True,4],"coordinate_space":"normalized_displacement_1000"}))
if __name__=="__main__":unittest.main()

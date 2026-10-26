"""H138 frozen model-native independent result admission tests."""
import json
import unittest
from h138.gpu_eval import SCHEMA_OUT,SOURCE_SHA,ADAPTER_SHA
from h138.owner_replay import verify,ARMS
from h138.probe import APPS
from h138.source import TITLES,SCHEMA,seed_columns,task_instruction

class FrozenFullAppNeuralTests(unittest.TestCase):
    def test_model_no_training_and_source_pins(self):
        self.assertEqual(len(SOURCE_SHA),64)
        self.assertEqual(len(ADAPTER_SHA),64)
        self.assertEqual(len(APPS),2)
        self.assertIn("real-neural",SCHEMA_OUT)

    def test_native_inference_falsification_rejected(self):
        cases=[]
        for app in APPS:
            for i,title in enumerate(TITLES):
                cases.append({"case_id":f"h138-{app}-{i}",
                 "app":app,"image_sha256":f"sha-{app}-{i}",
                 "start_bbox":{"x":95,"y":95,"width":10,"height":10},
                 "end_bbox":{"x":195,"y":195,"width":10,"height":10}})
        output=[]
        for arm in ARMS:
            for case in cases:
                for role,v in (("start",[104,139]),("end",[208,278])):
                    output.append({"arm":arm,"case_id":case["case_id"],
                      "image_sha256":case["image_sha256"],
                      "role":role,"raw":json.dumps({"point_2d":v}),
                      "point":[100,100] if role=="start" else [200,200],
                      "hit":True})
        receipt={"schema":SCHEMA_OUT,
           "status":"H138_PRIVATE_2_FULL_UPSTREAM_APPS_32_NEURAL_COMPLETE",
           "source_sha256":SOURCE_SHA,"adapter_sha256":ADAPTER_SHA,
           "source_app_revisions":APPS,"actual_neural_calls":32,
           "actual_optimizer_steps":0,"paid_compute":False,
           "main_Hala_flagship_weights_changed":False}
        self.assertEqual(len(verify({"cases":cases},receipt,output)),32)
        modified=[dict(x) for x in output]
        modified[0]["point"]=[700,400]
        with self.assertRaises(ValueError):
            verify({"cases":cases},receipt,modified)
        invalid=dict(receipt,actual_optimizer_steps=1)
        with self.assertRaises(ValueError):
            verify({"cases":cases},invalid,output)

if __name__=="__main__":unittest.main()

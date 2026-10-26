"""Frozen V3 DEV-only promotion gate cannot declare failed END geometry PASS."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h140.evaluate_v3_gate import evaluate, V2_STATUS, V3_STATUS
from h140.train_gpu_v3 import TRAIN_MANIFEST_SHA, DEV_MANIFEST_SHA

class H140V3DevPromotionGate(unittest.TestCase):
    def setup_receipt(self,path,v3,diverse,react_hits):
        path.mkdir()
        filename=("private-h140-v3-learning-evidence.json" if v3 else
                  "private-h140-learning-evidence.json")
        model_dir=path/("private-h140-v3-end-recovery-lora-rank8" if v3
                        else "private-h140-original-geometry-lora-rank8")
        model_dir.mkdir()
        raw=b"actual-small-test-weights"+str(v3).encode()
        (model_dir/"adapter_model.safetensors").write_bytes(raw)
        app_hits={
            "vanilla-native-board":{"both":11 if v3 else 10},
            "react-native-board":{"both":react_hits if v3 else 5, "end":react_hits if v3 else 5},
            "react-dndkit-board":{"both":10},
        }
        total={"start":62 if v3 else 64,
               "end":31 if v3 else 25,
               "both":30 if v3 else 25}
        receipt={
            "status":V3_STATUS if v3 else V2_STATUS,
            "real_optimizer_steps":160 if v3 else 128,
            "frozen_train_original_screens":256,
            "frozen_dev_original_screens":64,
            "train_manifest_sha256":TRAIN_MANIFEST_SHA,
            "dev_manifest_sha256":DEV_MANIFEST_SHA,
            "real_blind_neural_calls":0,
            "blind_gold_attached":False,"paid_compute_used":False,
            "flagship_Hala_35B_updated":False,
            "adapter_sha256":hashlib.sha256(raw).hexdigest(),
            "h140_adapter_dev":{
                "all_apps_and_roles_diverse":diverse,
                "point_hits":{"__all__":total,**app_hits}
            },
        }
        (path/filename).write_text(json.dumps(receipt),encoding="utf8")

    def test_no_blind_go_when_any_diversity_or_react_end_defect(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"v2",Path(directory)/"v3"
            self.setup_receipt(a,False,False,5)
            self.setup_receipt(b,True,False,9)
            outcome=evaluate(a,b)
            self.assertEqual(outcome["status"],"NOT_QUALIFIED_DO_NOT_TOUCH_BLIND")
            self.assertFalse(outcome["checks"]["original_3_app_end_diversity"])
            file=b/"private-h140-v3-learning-evidence.json"
            state=json.loads(file.read_text("utf8"))
            state["h140_adapter_dev"]["all_apps_and_roles_diverse"]=True
            state["h140_adapter_dev"]["point_hits"]["react-native-board"]["end"]=7
            state["h140_adapter_dev"]["point_hits"]["react-native-board"]["both"]=7
            file.write_text(json.dumps(state),encoding="utf8")
            self.assertFalse(evaluate(a,b)["checks"]["react_end_hits_at_least_8_of_22"])

    def test_all_predeclared_dev_gates_enable_single_offline_test_not_deploy(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"v2",Path(directory)/"v3"
            self.setup_receipt(a,False,False,5)
            self.setup_receipt(b,True,True,9)
            result=evaluate(a,b)
            self.assertEqual(result["status"],"ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND")
            self.assertEqual(result["v2_v3_independent_blind_calls"],0)
            self.assertFalse(result["flagship_Hala_35B_updated"])

if __name__=="__main__":
    unittest.main()

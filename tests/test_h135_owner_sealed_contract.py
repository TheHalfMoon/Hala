"""Strict independent native-response identity contract on frozen holdout."""
import copy
import unittest
from h135.owner_sealed import verify
from h135.sealed_gpu import (
    FORMAT,SEALED_MANIFEST_SHA,TRAINED_ADAPTER_SHA,
    SELECTED_MODE,SELECTED_DECISION_SHA)
from h132.grounder_contract import MODEL_ID,MODEL_SHA

class H135OwnerHoldoutContract(unittest.TestCase):
    def fixture(self):
        src={"cases":[]}
        rows=[]
        for i in range(18):
            id=f"h135-{620000+i}"
            src["cases"].append({
              "id":id,"image_sha256":f"im-{i}",
              "start_bbox":{"x":94,"y":94,"width":14,"height":14},
              "end_bbox":{"x":194,"y":194,"width":14,"height":14}})
        for arm in ("pretrained","h134_adapter"):
            for case in src["cases"]:
                for role,raw,xy in (("start",'{"point_2d":[104,139]}',[100,100]),
                                    ("end",'{"point_2d":[208,278]}',[200,200])):
                    rows.append({
                       "arm":arm,"case_id":case["id"],"image_sha256":case["image_sha256"],
                       "role":role,"raw":raw,"point":xy,"hit":True})
        report={"schema":FORMAT,
            "status":"H135_PRIVATE_ONCE_SEALED18_72_NEURAL_SCORED",
            "selected_resolution_mode":SELECTED_MODE,
            "frozen_pilot_decision_sha256":SELECTED_DECISION_SHA,
            "sealed_manifest_sha256":SEALED_MANIFEST_SHA,
            "immutable_adapter_sha256":TRAINED_ADAPTER_SHA,
            "frozen_base_id":MODEL_ID,"frozen_base_revision":MODEL_SHA,
            "actual_neural_calls":72,"optimizer_steps":0,
            "no_pilot_data_attached":True}
        return src,report,rows

    def test_raw_frozen_adapter_and_sample_identity(self):
        src,report,points=self.fixture()
        self.assertEqual(len(verify(src,report,points)),72)
        modified=copy.deepcopy(points)
        modified[0]["point"]=[4,2]
        with self.assertRaises(ValueError):verify(src,report,modified)
        modified=copy.deepcopy(points)
        modified[0]["image_sha256"]="wrong"
        with self.assertRaises(ValueError):verify(src,report,modified)
        modified=copy.deepcopy(report)
        modified["optimizer_steps"]=1
        with self.assertRaises(ValueError):verify(src,modified,points)
if __name__=="__main__":unittest.main()

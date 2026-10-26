"""Strict H136 once-only sealed24 neural-output provenance validation."""
import unittest
from h136.owner_sealed import verify_raw,ARMS
from h136.sealed_gpu import FORMAT,SEALED_MANIFEST_SHA,FROZEN_ADAPTER_SHA
from h136.train_gpu import BASE_MODEL,BASE_REV

class H136SealedContracts(unittest.TestCase):
    def test_exact_96_predictions_and_response_identity(self):
        scenes=[{"id":f"h136-{930000+i}","image_sha256":f"sha-{i}",
                 "start_bbox":{"x":95,"y":95,"width":10,"height":10},
                 "end_bbox":{"x":195,"y":195,"width":10,"height":10}}
                for i in range(24)]
        rows=[]
        for arm in ARMS:
            for scene in scenes:
                for role,p in (("start",[100,100]),("end",[200,200])):
                    normalized=[round(p[0]/960*1000),round(p[1]/720*1000)]
                    import json
                    rows.append({"arm":arm,"case_id":scene["id"],
                                 "image_sha256":scene["image_sha256"],
                                 "role":role,"raw":json.dumps({"point_2d":normalized}),
                                 "point":p,"hit":True})
        info={"schema":FORMAT,"status":"H136_REAL_ONCE_ONLY_SEALED24_NEURAL_COMPLETE",
              "frozen_original_source_manifest_sha256":SEALED_MANIFEST_SHA,
              "frozen_adapter_sha256":FROZEN_ADAPTER_SHA,
              "base_model":BASE_MODEL,"base_revision":BASE_REV,
              "sealed_new_scene_count":24,"neural_inference_calls":96,
              "actual_optimizer_steps":0,"train_or_dev_images_attached":False,
              "paid_compute_used":False}
        result=verify_raw({"cases":scenes},info,rows)
        self.assertEqual(len(result),96)
        changed=[dict(x) for x in rows]
        changed[0]["point"]=[0,0]
        with self.assertRaises(ValueError):verify_raw({"cases":scenes},info,changed)
        altered=dict(info)
        altered["actual_optimizer_steps"]=1
        with self.assertRaises(ValueError):verify_raw({"cases":scenes},altered,rows)

if __name__=="__main__":unittest.main()

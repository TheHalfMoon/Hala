"""Synthetic-only independent neural response/source replay regression."""
import copy
import unittest
from h128.owner_replay import replay
from h128.zoom import anchored_crop,Crop
from h125.model_contracts import MODELS
from h128.source import FAMILIES

class H128ReplayTests(unittest.TestCase):
    def fixture(self):
        source={"schema":"hala.h128.owner-private-true-original-TRAIN16-source-replay.v1",
                "source_schedule_sha256":"a"*64,
                "source_cases_sha256":"b"*64,
                "original_image_byte_sha_verified":16,
                "original_previous_excluded":2688,"cases":[]}
        raw=[]
        for i in range(16):
            id=f"synthetic-{i}"
            sha=f"synthetic-sha-{i}"
            source["cases"].append({
                "id":id,"image_sha256":sha,"family":FAMILIES[i//4],
                "image_width":800,"image_height":600,
                "start_bbox":[97,97,105,105],"end_bbox":[597,397,605,405]})
            raw.append({
                "id":id,"image_sha256":sha,
                "direct_raw_start":"[0.125,0.1666666667]",
                "direct_start":[100,100],
                "crop_start":[0,0,400,300],
                "zoom_raw_start":"[0.25,0.3333333333]",
                "zoom_start":[100,100],
                "direct_raw_end":"[0.75,0.6666666667]",
                "direct_end":[600,400],
                "crop_end":[400,250,800,550],
                "zoom_raw_end":"[0.5,0.5]",
                "zoom_end":[600,400]
            })
        from h128.zoom import score
        pairs=[{"id":c["id"],"image_sha256":c["image_sha256"],
                "direct_start":(100,100),"zoom_start":(100,100),
                "crop_start":Crop(0,0,400,300),
                "direct_end":(600,400),"zoom_end":(600,400),
                "crop_end":Crop(400,250,800,550)}
               for c in source["cases"]]
        scored=score(source["cases"],pairs)
        reported={"schedule_sha256":source["source_schedule_sha256"],
                  "source_cases_sha256":source["source_cases_sha256"],
                  "model_id":MODELS["showui_2b"].model_id,
                  "model_revision":MODELS["showui_2b"].revision_sha,
                  "real_neural_calls":64,"private_raw_sha256":"c"*64,
                  "aggregate":scored["aggregate"],"per_family":scored["per_family"]}
        return source,raw,reported

    def test_real_native_contract_replays_score_without_oracle_in_crop(self):
        source,raw,reported=self.fixture()
        receipt=replay(source,raw,reported,"c"*64)
        self.assertEqual(receipt["real_model_calls_verified"],64)
        self.assertEqual(receipt["independent_metric_recompute"]["zoom_full"],16)
        self.assertFalse(receipt["model_skill_qualified"])

    def test_refuses_raw_response_coordinate_or_provenance_tamper(self):
        source,raw,reported=self.fixture()
        for field,value in (("direct_raw_start","[0.3,0.3]"),
                            ("crop_end",[0,0,400,300]),
                            ("zoom_raw_end","[0.1,0.1]")):
            corrupt=copy.deepcopy(raw)
            corrupt[0][field]=value
            with self.assertRaises(ValueError):
                replay(source,corrupt,reported,"c"*64)
        with self.assertRaises(ValueError):
            replay(source,raw,reported,"d"*64)
        broken=copy.deepcopy(reported)
        broken["aggregate"]["zoom_full"]=0
        with self.assertRaises(ValueError):
            replay(source,raw,broken,"c"*64)

if __name__=="__main__":unittest.main()

"""H129 independent source/native-output checksum replay synthetic tests."""
import copy
import unittest
from h125.model_contracts import MODELS
from h129.feedback import score
from h129.owner_replay import replay
from h129.kaggle_private_source_run import qualifies_source_receipt
from h129.source import FAMILIES

def data():
    source={"schema":"hala.h129.owner-private-true-original-TRAIN16-source-replay.v1",
            "source_schedule_sha256":"a"*64,"source_cases_sha256":"b"*64,
            "original_image_byte_sha_verified":16,"original_previous_excluded":2704,
            "cases":[]}
    responses=[]
    for i in range(16):
        source["cases"].append({"id":f"sample-{i}","image_sha256":f"sha-{i}",
                                "family":FAMILIES[i//4],"image_width":800,
                                "image_height":600,
                                "start_bbox":[98,98,104,104],
                                "end_bbox":[598,398,604,404]})
        responses.append({"id":f"sample-{i}","image_sha256":f"sha-{i}",
                          "direct_raw_start":"[0.125,0.1666666667]",
                          "feedback_raw_start":"[0.125,0.1666666667]",
                          "direct_start":[100,100],"feedback_start":[100,100],
                          "direct_raw_end":"[0.75,0.6666666667]",
                          "feedback_raw_end":"[0.75,0.6666666667]",
                          "direct_end":[600,400],"feedback_end":[600,400]})
    pred=[{"id":c["id"],"image_sha256":c["image_sha256"],
           "direct_start":(100,100),"feedback_start":(100,100),
           "direct_end":(600,400),"feedback_end":(600,400)}
          for c in source["cases"]]
    metrics=score(source["cases"],pred)
    reported={"source_schedule_sha256":source["source_schedule_sha256"],
              "source_cases_sha256":source["source_cases_sha256"],
              "model_id":MODELS["showui_2b"].model_id,
              "model_revision":MODELS["showui_2b"].revision_sha,
              "real_neural_calls":64,"private_raw_sha256":"c"*64,
              "is_real_interactive_environment":False,
              "aggregate":metrics["aggregate"],"per_family":metrics["per_family"]}
    return source,responses,reported

class H129OwnerTests(unittest.TestCase):
    def test_cpu_source_receipt_rejects_old_2688_count(self):
        result={"status":"PASS_H129_OWNER_PRIVATE_TRAIN16_SOURCE_REPLAY_CPU",
                "source_verified":16,"excluded_previous":2704,
                "model_inference_calls":0,"optimizer_steps":0}
        self.assertTrue(qualifies_source_receipt(result))
        result["excluded_previous"]=2688
        self.assertFalse(qualifies_source_receipt(result))

    def test_exact_private_native_replay(self):
        source,responses,metrics=data()
        result=replay(source,responses,metrics,"c"*64)
        self.assertEqual(result["verified_model_calls"],64)
        self.assertEqual(result["independent_recomputed_metrics"]["feedback_full"],16)
        self.assertFalse(result["model_skill_qualified"])

    def test_tampering_refused(self):
        source,responses,metrics=data()
        for column,evil in [("direct_raw_start","[0.2,0.2]"),
                            ("feedback_raw_end","[0.2,0.2]"),
                            ("feedback_end",[1,2])]:
            altered=copy.deepcopy(responses)
            altered[0][column]=evil
            with self.assertRaises(ValueError):
                replay(source,altered,metrics,"c"*64)
        with self.assertRaises(ValueError):
            replay(source,responses,metrics,"0"*64)
        reported=copy.deepcopy(metrics)
        reported["aggregate"]["feedback_full"]=0
        with self.assertRaises(ValueError):
            replay(source,responses,reported,"c"*64)

if __name__=="__main__": unittest.main()

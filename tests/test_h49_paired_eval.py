"""CPU-only fabricated fixture tests: NEVER actual model results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from h49.factory import canonical, sha
from h49.audit import inspect
from h49.paired_eval import (SCHEMA, MODEL, REVISION, digest_prompt,
                             parse_action, score)

ROOT = Path("h49/assets")


def click(x, y):
    return json.dumps({"type":"click","point":[int(x),int(y)],
        "coordinate_space":"window","image_scale":1,
        "element_ref_hint":"visual_target"}, sort_keys=True)


class PairedScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.manifest=inspect(ROOT)
        model=json.loads((ROOT/"model_inputs.json").read_text("utf-8"))
        oracle=json.loads((ROOT/"evaluator_only.json").read_text("utf-8"))
        cls.dev=[r for r in model["cases"] if r["split"]=="dev"]
        cls.oracle={r["id"]:r for r in oracle["cases"] if r["split"]=="dev"}

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.rows,self.meta,self.hashes={},{},{}
        for arm in ("baseline","adapter"):
            self.rows[arm]=[]
            for row in self.dev:
                x,y,w,h=self.oracle[row["id"]]["target_bounds"]
                self.rows[arm].append({
                    "case_id":row["id"],"language":row["language"],
                    "image_sha256":row["image_sha256"],"arm":arm,
                    "raw":click(x+w/2,y+h/2) if arm=="adapter" else click(0,0),
                    "error":None,"elapsed_seconds":1.25,"output_tokens":30})
            self.meta[arm]={
                "schema":SCHEMA,"arm":arm,"model_id":MODEL,
                "model_revision":REVISION,"processor_revision":REVISION,
                "prompt_sha256":digest_prompt([r["goal"] for r in self.dev]),
                "source_sha256":self.manifest["source_sha256"],
                "capture_sha256":self.manifest["capture_sha256"],
                "dataset_sha256":self.manifest["dataset_sha256"],
                "gpu_name":"UNIT_TEST_NOT_REAL_GPU","precision":"fp16",
                "provider":"colab","provider_run_id":"UNIT_TEST_NOT_REAL",
                "adapter_sha256":None if arm=="baseline" else "a"*64,
                "raw_sha256":None,"status":"completed"}
        self.flush()

    def flush(self):
        for arm in ("baseline","adapter"):
            raw="".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                        for r in self.rows[arm]).encode("utf-8")
            (self.path/(arm+".jsonl")).write_bytes(raw)
            self.meta[arm]["raw_sha256"]=sha(raw)
            meta=canonical(self.meta[arm])
            (self.path/(arm+".json")).write_bytes(meta)
            self.hashes[arm]=(sha(meta),sha(raw))

    def evaluate(self,**update):
        args=dict(root=ROOT,baseline_meta=self.path/"baseline.json",
            baseline_raw=self.path/"baseline.jsonl",
            baseline_meta_sha=self.hashes["baseline"][0],
            baseline_raw_sha=self.hashes["baseline"][1],
            adapter_meta=self.path/"adapter.json",
            adapter_raw=self.path/"adapter.jsonl",
            adapter_meta_sha=self.hashes["adapter"][0],
            adapter_raw_sha=self.hashes["adapter"][1])
        args.update(update)
        return score(**args)

    def test_fake_pair_classification_is_never_model_qualification(self):
        report=self.evaluate()
        self.assertFalse(report["real_model_provenance_independently_verified"])
        self.assertFalse(report["independent_blind_test"])
        self.assertFalse(report["quality_qualified"])
        self.assertEqual(report["paired"]["all"]["denominator"],32)
        self.assertEqual(report["paired"]["all"]["adapter_only_correct"],32)
        self.assertEqual(report["paired"]["ar"]["adapter_only_correct"],16)
        self.assertEqual(report["paired"]["en"]["adapter_only_correct"],16)
        self.assertEqual(report["counts"]["baseline"]["all"]["unbound"],32)
        self.assertEqual(report["counts"]["adapter"]["all"]["correct"],32)

    def test_wrong_target_not_correct(self):
        capture=json.loads((ROOT/"capture.json").read_text("utf-8"))
        record=next(x for x in capture["records"] if x["id"]==self.dev[0]["id"])
        oracle=self.oracle[self.dev[0]["id"]]
        wrong=next(x for x in record["controls"] if x["key"]!=oracle["target_key"])
        x,y,w,h=wrong["bounds"]
        self.rows["adapter"][0]["raw"]=click(x+w/2,y+h/2)
        self.flush()
        result=self.evaluate()
        self.assertEqual(result["per_case"][0]["adapter"],"wrong_target")
        self.assertEqual(result["paired"]["all"]["adapter_only_correct"],31)

    def test_abstain_invalid_and_inference_error(self):
        self.rows["adapter"][0]["raw"]=json.dumps({"abstain":True})
        self.rows["adapter"][1]["raw"]=json.dumps({"point":[0,0]})
        self.rows["adapter"][2]["raw"]=None
        self.rows["adapter"][2]["error"]="OutOfMemory"
        self.flush()
        r=self.evaluate()
        self.assertEqual([x["adapter"] for x in r["per_case"][:3]],
                         ["abstain","invalid_json_or_schema","inference_error"])

    def test_reject_bad_external_hash(self):
        with self.assertRaises(ValueError):
            self.evaluate(adapter_raw_sha="0"*64)
        path=self.path/"adapter.jsonl"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_reject_missing_duplicate_and_wrong_order(self):
        for what in ("missing","duplicate","reordered","changed_image"):
            with self.subTest(what=what):
                saved=copy.deepcopy(self.rows["adapter"])
                if what=="missing":
                    self.rows["adapter"].pop()
                elif what=="duplicate":
                    self.rows["adapter"][1]=copy.deepcopy(self.rows["adapter"][0])
                elif what=="reordered":
                    self.rows["adapter"][0],self.rows["adapter"][1]=self.rows["adapter"][1],self.rows["adapter"][0]
                else:
                    self.rows["adapter"][0]["image_sha256"]="0"*64
                self.flush()
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.rows["adapter"]=saved

    def test_reject_mismatched_hardware_source_adapter_and_prompt(self):
        for field,bad in (("gpu_name","other"),("provider_run_id","different_session"),("source_sha256","0"*64),
                          ("adapter_sha256",None),("model_revision","f"*40),
                          ("prompt_sha256","f"*64),("precision","bf16"),
                          ("provider","kaggle"),("status","incomplete")):
            with self.subTest(field=field):
                original=self.meta["adapter"][field]
                self.meta["adapter"][field]=bad
                self.flush()
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.meta["adapter"][field]=original

    def test_strict_json_action_and_no_fence_or_partial_schema(self):
        self.assertEqual(parse_action(click(100,200)),(100,200))
        self.assertIsNone(parse_action(json.dumps({"abstain":True})))
        for raw in (
            json.dumps({"point":[100,200]}),
            '{"abstain":false}', '{"point": [NaN, 2]}',
            '{"type":"click","type":"click"}',
            click(10000,1),
            click(1,2).replace('"image_scale": 1','"image_scale": true'),
            click(1,2).replace('[1, 2]','[1.5, 2]'),
        ):
            with self.subTest(raw=raw[:50]), self.assertRaises(ValueError):
                parse_action(raw)

    def test_only_external_run_can_be_actual_model_evidence(self):
        report=self.evaluate()
        self.assertFalse(report["real_model_provenance_independently_verified"])
        self.assertTrue(report["public_dev_only"])
        self.assertEqual(len(report["per_case"]),32)

if __name__=="__main__":
    unittest.main()

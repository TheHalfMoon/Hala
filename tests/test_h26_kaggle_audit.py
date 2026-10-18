"""H26 CPU notebook submission guards + independent raw audit fault injection."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h26 import runner,choice
from tools import build_h26_kaggle,verify_h26_dev
from h24.mind_dev import MODEL_ID,MODEL_REVISION

SHA="a"*40


def fake_raw():
    out=[]
    for row in runner.validated_rows():
        ref=row["source_ref"]
        code=row["private_reference"]["choice"]
        record={"id":ref["id"],"family":ref["family"],
                "language":ref["language"],"state":ref["state"],
                "raw":code,"error":None,"wall_seconds":0.4}
        record.update(choice.score(code,row["private_reference"]["typed"]))
        out.append(record)
    return {
        "schema":runner.SCHEMA,"status":"measured_unqualified",
        "protocol":runner.RUN_PROTOCOL,"source_commit":SHA,
        "data_sha256":runner.EXACT_DATA_SHA,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "gpu_name":"Tesla T4","provider":"kaggle","precision":"fp16",
        "partition":"dev","training_runs":0,
        "native_device_actions_executed":False,"model_adopted":False,
        "load_seconds":23.0,"torch_allocated_after_load_bytes":3500000000,
        "torch_peak_allocated_bytes":3800000000,
        "outcomes":out,"summary":runner.aggregate(out),
    }


class H26NotebookTests(unittest.TestCase):
    def test_private_exact_source_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"fresh"
            result=build_h26_kaggle.generate(SHA,folder)
            self.assertFalse(result["submitted"])
            self.assertFalse(result["model_inference_performed"])
            metadata=json.loads((folder/"kernel-metadata.json").read_text("utf8"))
            notebook=json.loads((folder/"hala-h26.ipynb").read_text("utf8"))
            self.assertTrue(metadata["is_private"] and metadata["enable_gpu"] and metadata["enable_internet"])
            self.assertFalse(metadata["enable_tpu"])
            self.assertEqual(metadata["dataset_sources"],[])
            self.assertEqual(notebook["metadata"]["hala_protocol"],runner.RUN_PROTOCOL)
            self.assertEqual(notebook["metadata"]["hala_source_commit"],SHA)
            self.assertEqual(notebook["metadata"]["hala_split"],"dev")
            self.assertEqual(len(notebook["cells"]),5)
            text="".join("".join(c["source"]) for c in notebook["cells"])
            self.assertIn("rev-parse",text)
            self.assertIn("h26.runner",text)
            self.assertIn("HALA_REMOTE_DEVELOPMENT_ALLOWED",text)
            self.assertNotIn("--split test",text.lower())
            with self.assertRaises(FileExistsError):
                build_h26_kaggle.generate(SHA,folder)

    def test_new_reviewed_sha40_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("main","ps/h26-typed-choice","0"*39,"A"*40):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    build_h26_kaggle.generate(value,Path(tmp)/value.replace("/","-"))
            self.assertFalse(any(Path(tmp).iterdir()))


class H26AuditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file=Path(self.tmp.name)/"result.json"
        self.original=fake_raw()

    def inspect(self,value=None,digest=None):
        content=(json.dumps(value if value is not None else self.original,
                            sort_keys=True,ensure_ascii=False)+"\n").encode("utf8")
        self.file.write_bytes(content)
        return verify_h26_dev.inspect(self.file,digest or hashlib.sha256(content).hexdigest(),SHA)

    def test_perfect_oracle_fake_is_not_actual_model_quality(self):
        result=self.inspect()
        self.assertEqual(result["language_state_metrics"]["all"]["semantic_correct"],42)
        self.assertFalse(result["hardware_attested_independently"])
        self.assertFalse(result["production_model_approved"])

    def test_wrong_sha_never_accepted(self):
        with self.assertRaises(ValueError):
            self.inspect(digest="0"*64)

    def test_change_claims_fail_closed(self):
        for key,val in (
            ("training_runs",1),("model_adopted",True),
            ("native_device_actions_executed",True),
            ("data_sha256","0"*64),("model_id","unapproved"),
            ("partition","test"),("source_commit","b"*40),
            ("status","running"),("gpu_name","H100"),
        ):
            changed=copy.deepcopy(self.original)
            changed[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.inspect(changed)

    def test_tampered_case_order_or_reference_refused(self):
        for mutator in (
            lambda x:x["outcomes"].pop(),
            lambda x:x["outcomes"].reverse(),
            lambda x:x["outcomes"][0].update({"semantic_correct":False}),
            lambda x:x["outcomes"][0].update({"raw":"C7"}),
            lambda x:x["outcomes"][0].update({"state":"missing"}),
            lambda x:x["summary"]["all"].update({"semantic_correct":43}),
        ):
            changed=copy.deepcopy(self.original)
            mutator(changed)
            with self.assertRaises(ValueError):
                self.inspect(changed)

    def test_valid_code_not_always_semantically_correct(self):
        changed=copy.deepcopy(self.original)
        case=runner.validated_rows()[0]
        changed["outcomes"][0].update(raw="C0",**choice.score("C0",case["private_reference"]["typed"]))
        changed["summary"]=runner.aggregate(changed["outcomes"])
        report=self.inspect(changed)
        self.assertEqual(report["language_state_metrics"]["all"]["semantic_correct"],41)
        self.assertEqual(report["language_state_metrics"]["all"]["withheld"],1)

    def test_exception_requires_proof_and_cannot_inflate_results(self):
        changed=copy.deepcopy(self.original)
        changed["outcomes"][0].update(
            raw=None,error="model_inference_error",error_type="RuntimeError",
            format_valid=False,withheld=False,semantic_correct=False,suggestion=None)
        changed["summary"]=runner.aggregate(changed["outcomes"])
        report=self.inspect(changed)
        self.assertEqual(report["language_state_metrics"]["all"]["inference_errors"],1)
        changed["outcomes"][0].pop("error_type")
        with self.assertRaises(ValueError):
            self.inspect(changed)


if __name__=="__main__":
    unittest.main()

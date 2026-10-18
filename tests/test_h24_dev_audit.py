"""H24 Kaggle DEV evidence independent CPU fault-injection checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h24 import mind_dev
from tools import build_h24_kaggle, verify_h24_dev


SOURCE="a"*40


def fake_measured():
    cases=mind_dev.dev_cases()
    rows=[]
    for c in cases:
        raw=json.dumps(c["expected"])
        record={"id":c["id"],"family":c["family"],"language":c["language"],
                "variant":c["variant"],"raw":raw,"error":None,
                "wall_seconds":0.5}
        record.update(mind_dev.score(c,raw))
        rows.append(record)
    return {"schema":"hala.h24.mind_dev_baseline.v1",
        "status":"measured_unqualified","model_id":mind_dev.MODEL_ID,
        "model_revision":mind_dev.MODEL_REVISION,
        "source_commit":SOURCE,"frozen_h24_sha256":mind_dev.H24_CASES_SHA256,
        "provider":"kaggle","precision":"fp16","split":"dev",
        "training_runs":0,"model_adopted":False,"native_qualified":False,
        "task_cases":20,"gpu_name":"Tesla T4",
        "load_seconds":8.5,"cuda_allocated_after_load_bytes":4000000000,
        "cuda_peak_allocated_bytes":5000000000,
        "outcomes":rows,
        "summary":mind_dev.summary(rows)}


class H24DevelopmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"result.json"
        self.original=fake_measured()
        self.save(self.original)

    def save(self,data):
        raw=(json.dumps(data,ensure_ascii=False,sort_keys=True)+"\n").encode("utf-8")
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def audit(self, data=None):
        if data is None:
            data=self.original
        h=self.save(data)
        return verify_h24_dev.verify(self.path,h,SOURCE)

    def test_full_20_case_raw_rescoring(self):
        r=self.audit()
        self.assertTrue(r["raw_reply_scoring_verified"])
        self.assertEqual(r["strata"]["all"]["cases"],20)
        self.assertEqual(r["strata"]["all"]["exact_correct"],20)
        self.assertFalse(r["model_qualified"])

    def test_exact_external_hash_mandatory(self):
        with self.assertRaises(ValueError):
            verify_h24_dev.verify(self.path,"0"*64,SOURCE)
        with self.assertRaises(ValueError):
            verify_h24_dev.verify(self.path,"NOT_SHA",SOURCE)

    def test_tampered_score_fails(self):
        report=copy.deepcopy(self.original)
        report["outcomes"][0]["tool_correct"]=False
        with self.assertRaises(ValueError):
            self.audit(report)

    def test_false_model_adoption_fails(self):
        report=copy.deepcopy(self.original)
        report["model_adopted"]=True
        with self.assertRaises(ValueError):
            self.audit(report)

    def test_missing_and_repeated_rows_fail(self):
        report=copy.deepcopy(self.original)
        report["outcomes"].pop()
        with self.assertRaises(ValueError):
            self.audit(report)
        report=copy.deepcopy(self.original)
        report["outcomes"][1]["id"]=report["outcomes"][0]["id"]
        with self.assertRaises(ValueError):
            self.audit(report)

    def test_fake_test_case_fail_closed(self):
        report=copy.deepcopy(self.original)
        report["outcomes"][0]["family"]="library_lending"
        with self.assertRaises(ValueError):
            self.audit(report)

    def test_wrong_mapped_language_and_summary_fail(self):
        report=copy.deepcopy(self.original)
        report["outcomes"][0]["language"]="ar" if report["outcomes"][0]["language"]=="en" else "en"
        with self.assertRaises(ValueError):
            self.audit(report)
        report=copy.deepcopy(self.original)
        report["summary"]["all"]["exact_correct"]=19
        with self.assertRaises(ValueError):
            self.audit(report)

    def test_inference_error_counted_not_success(self):
        report=copy.deepcopy(self.original)
        row=report["outcomes"][0]
        row.update(raw=None,valid=False,tool_correct=False,exact_correct=False,
                   predicted_tool=None,error="model_inference_error",
                   error_type="RuntimeError")
        report["summary"]=mind_dev.summary(report["outcomes"])
        result=self.audit(report)
        self.assertEqual(result["strata"]["all"]["inference_errors"],1)
        self.assertEqual(result["strata"]["all"]["exact_correct"],19)

    def test_generator_requires_immutable_commit_and_new_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"out"
            with self.assertRaises(ValueError):
                build_h24_kaggle.generate("main",folder)
            result=build_h24_kaggle.generate(SOURCE,folder)
            self.assertTrue(Path(result["notebook"]).exists())
            data=json.loads(Path(result["notebook"]).read_text(encoding="utf-8"))
            kernel=json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            self.assertTrue(kernel["is_private"] and kernel["enable_gpu"])
            self.assertTrue(kernel["enable_internet"])
            self.assertEqual(data["metadata"]["hala_h24_split"],"dev")
            source="".join(data["cells"][-1]["source"])
            self.assertIn("h24.mind_dev",source)
            self.assertIn(SOURCE,source)
            self.assertNotIn("--split test",source)
            with self.assertRaises(FileExistsError):
                build_h24_kaggle.generate(SOURCE,folder)

    def test_no_unpinned_remote_workflows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/"build"
            build_h24_kaggle.generate(SOURCE,folder)
            n=json.loads((folder/"h24-mind-dev.ipynb").read_text(encoding="utf-8"))
            self.assertEqual(len(n["cells"]),5)
            self.assertIn("HALA_REMOTE_DEVELOPMENT_ALLOWED", "".join(n["cells"][3]["source"]))
            self.assertIn("KAGGLE_KERNEL_RUN_TYPE", "".join(n["cells"][1]["source"]))


if __name__=="__main__":
    unittest.main()

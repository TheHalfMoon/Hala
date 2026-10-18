"""Independent H24 diagnostic regression, distinguishing JSON vs typed acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h24.mind_dev import dev_cases, score, summary
from tests.test_h24_dev_audit import fake_measured, SOURCE
from tools.diagnose_h24_dev import inspect


class H24DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.file=Path(self.temp.name)/"result.json"
        self.data=fake_measured()

    def inspect(self):
        raw=(json.dumps(self.data,sort_keys=True,ensure_ascii=False)+"\n").encode("utf-8")
        self.file.write_bytes(raw)
        return inspect(self.file,hashlib.sha256(raw).hexdigest(),SOURCE)

    def test_correct_model_response_is_also_schema_valid(self):
        d=self.inspect()
        self.assertEqual(d["total"]["json_syntax_valid"],20)
        self.assertEqual(d["total"]["strict_typed_valid"],20)
        self.assertEqual(d["total"]["tool_correct_from_raw_untrusted"],20)

    def test_free_text_subgoal_not_executable_but_json_syntax_valid(self):
        for case,record in zip(dev_cases(),self.data["outcomes"]):
            raw=json.dumps({"kind":"next","tool":case["expected"]["tool"],
                            "subgoal":"Click a button with a natural-language instruction."})
            record["raw"]=raw
            record.update(score(case,raw))
        self.data["summary"]=summary(self.data["outcomes"])
        result=self.inspect()
        self.assertEqual(result["total"]["json_syntax_valid"],20)
        self.assertEqual(result["total"]["three_fields_valid"],20)
        self.assertEqual(result["total"]["strict_typed_valid"],0)
        self.assertEqual(result["total"]["tool_correct_from_raw_untrusted"],20)
        self.assertTrue(all(not c["strict_typed_valid"] for c in result["cases"]))

    def test_wrong_visual_on_unsafe_user_request_is_counted_not_executed(self):
        index=next(i for i,c in enumerate(dev_cases()) if c["variant"]=="unsafe_request")
        case=dev_cases()[index]
        raw=json.dumps({"kind":"next","tool":"visual","subgoal":"click_instead"})
        self.data["outcomes"][index]["raw"]=raw
        self.data["outcomes"][index].update(score(case,raw))
        self.data["summary"]=summary(self.data["outcomes"])
        report=self.inspect()
        self.assertEqual(report["total"]["visual_on_unsafe_user_case"],1)
        self.assertEqual(report["total"]["tool_correct_from_raw_untrusted"],19)
        self.assertFalse(report["cases"][index]["tool_correct_from_raw_untrusted"])
        self.assertFalse(report["no_model_adoption"] is False)

    def test_hash_mismatch_fails_before_any_diagnostics(self):
        self.file.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):
            inspect(self.file,"0"*64,SOURCE)


if __name__=="__main__":
    unittest.main()

"""CPU-only H26 42-case choice scorer + fail-closed remote admission tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from h26 import runner,choice,corpus


def fabricated_oracle_only():
    result=[]
    for r in runner.validated_rows():
        ref=r["source_ref"]
        code=r["private_reference"]["choice"]
        record={"id":ref["id"],"family":ref["family"],"language":ref["language"],
                "state":ref["state"],"raw":code,"error":None}
        record.update(choice.score(code,r["private_reference"]["typed"]))
        result.append(record)
    return result


class H26RunnerTests(unittest.TestCase):
    def test_exact_42_prospective_cases(self):
        rows=runner.validated_rows()
        self.assertEqual(len(rows),42)
        self.assertEqual(corpus.validate(corpus.dataset())["sha256"],runner.EXACT_DATA_SHA)

    def test_fabricated_source_oracle_only_for_testing_scoring(self):
        records=fabricated_oracle_only()
        scores=runner.aggregate(records)
        self.assertEqual(scores["all"]["semantic_correct"],42)
        self.assertEqual(scores["ar"]["semantic_correct"],21)
        self.assertEqual(scores["en"]["semantic_correct"],21)
        self.assertEqual(scores["all"]["withheld"],0)
        self.assertEqual(scores["all"]["unsafe_user_misrouted"],0)

    def test_format_valid_is_not_semantic_correctness(self):
        rows=fabricated_oracle_only()
        rows[0].update(choice.score("C0",runner.validated_rows()[0]["private_reference"]["typed"]),raw="C0")
        result=runner.aggregate(rows)
        self.assertEqual(result["all"]["complete_codes"],42)
        self.assertEqual(result["all"]["semantic_correct"],41)
        self.assertEqual(result["all"]["withheld"],1)

    def test_unsafe_user_semantic_misrouting_exposed(self):
        rows=fabricated_oracle_only()
        index=next(i for i,x in enumerate(rows) if x["state"]=="unsafe_user")
        rows[index].update(choice.score("C1",runner.validated_rows()[index]["private_reference"]["typed"]),raw="C1")
        result=runner.aggregate(rows)
        self.assertEqual(result["all"]["unsafe_user_misrouted"],1)
        self.assertEqual(result["all"]["semantic_correct"],41)

    def test_missing_inference_or_wrong_error_flags_rejected(self):
        rows=fabricated_oracle_only()
        rows[0].update(choice.score("C0",runner.validated_rows()[0]["private_reference"]["typed"]),
                       raw=None,error="model_inference_error",error_type="RuntimeError")
        rows[0].update(format_valid=False,withheld=False,semantic_correct=False,suggestion=None)
        self.assertEqual(runner.aggregate(rows)["all"]["inference_errors"],1)
        rows[0]["error_type"] = None
        with self.assertRaises(ValueError):
            runner.aggregate(rows)

    def test_duplicated_row_missing_row_and_forged_score_denied(self):
        for mutator in (
            lambda x:x.pop(),
            lambda x:x[0].update({"id":x[1]["id"]}),
            lambda x:x[0].update({"semantic_correct":False}),
            lambda x:x[0].update({"state":"unsafe_user"}),
            lambda x:x[0].update({"raw":"C2"}),
            lambda x:x[0].update({"family":"museum_accessions"}),
            lambda x:x[0].update({"suggestion":{"kind":"next","tool":"shell","subgoal":"run_os"}}),
        ):
            data=fabricated_oracle_only()
            mutator(data)
            with self.assertRaises(ValueError):
                runner.aggregate(data)

    def test_unexpected_prose_is_invalid_not_typed_success(self):
        records=fabricated_oracle_only()
        records[0].update(raw="C7 because forbidden",format_valid=False,
                          withheld=False,semantic_correct=False,suggestion=None)
        aggregate=runner.aggregate(records)
        self.assertEqual(aggregate["all"]["complete_codes"],41)
        self.assertEqual(aggregate["all"]["semantic_correct"],41)

    def test_windows_cannot_allocate_kaggle_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"remote"
            with self.assertRaises(RuntimeError):
                runner.remote_admit("a"*40,output)
            self.assertFalse(output.exists())

    def test_no_gpu_models_imported_in_cpu_suite(self):
        self.assertFalse(hasattr(runner,"torch"))
        self.assertFalse(hasattr(runner,"model"))
        self.assertFalse(hasattr(runner,"processor"))

    def test_no_old_test_families_or_reference_in_model_prompt(self):
        for row in runner.validated_rows():
            model=json.dumps(row["model_input"],ensure_ascii=False)
            self.assertNotIn("source_ref",model)
            self.assertNotIn("private_reference",model)
            self.assertNotIn(row["source_ref"]["id"],model)


if __name__=="__main__":
    unittest.main()

"""Source-only H27 H26 post-hoc diagnostic tests; NO model performance claims."""
import copy
import tempfile
from pathlib import Path
import unittest

from h26 import choice, runner
from tools.analyze_h26_failure import analyze, describe


def synthetic_rows(select):
    """Synthetic hand-coded choices are NOT Kaggle outcomes or held-out TEST."""
    output = []
    for case in runner.validated_rows():
        info = case["source_ref"]
        code = select(info)
        row = {
            "id": info["id"], "language": info["language"],
            "family": info["family"], "state": info["state"],
            "raw": code, "error": None, "error_type": None,
        }
        row.update(choice.score(code, case["private_reference"]["typed"]))
        output.append(row)
    return output


class H27FailureForensicsTests(unittest.TestCase):
    def test_all_c1_detected_not_reported_as_accuracy(self):
        rows = synthetic_rows(lambda _: "C1")
        result = describe(rows)
        self.assertEqual(result["by_language"]["all"]["c1_selected"], 42)
        self.assertEqual(result["by_language"]["all"]["c1_correct"], 6)
        self.assertEqual(result["by_language"]["en"]["c1_selected"], 21)
        self.assertEqual(result["by_language"]["ar"]["c1_selected"], 21)
        self.assertEqual(result["by_language"]["all"]["unsafe_user_wrong_proposal"], 6)
        self.assertEqual(result["by_state"]["stale"]["correct"], 0)
        self.assertFalse(result["causal_root_cause_established"])

    def test_c0_withholding_never_counts_as_explicit_refusal(self):
        rows = synthetic_rows(lambda x: "C0" if x["state"] == "unsafe_user" else "C1")
        report = describe(rows)
        self.assertEqual(report["by_language"]["all"]["unsafe_user_withheld_c0"], 6)
        self.assertEqual(report["by_language"]["all"]["unsafe_user_explicit_c7"], 0)
        self.assertEqual(report["by_language"]["all"]["unsafe_user_wrong_proposal"], 0)
        self.assertEqual(report["by_language"]["all"]["c0_withheld"], 6)
        self.assertFalse(report["model_or_release_admissible"])

    def test_correct_refusal_and_withholding_distinct(self):
        rows = synthetic_rows(lambda x: "C7" if x["state"] == "unsafe_user" else "C0")
        report = describe(rows)
        self.assertEqual(report["by_language"]["all"]["unsafe_user_explicit_c7"], 6)
        self.assertEqual(report["by_language"]["all"]["unsafe_user_withheld_c0"], 0)
        self.assertEqual(report["by_language"]["all"]["typed_proposals"], 6)
        self.assertEqual(report["by_language"]["all"]["c0_withheld"], 36)
        self.assertFalse(report["confidence_calibrated"])

    def test_confusion_supports_codes_and_source_oracle_only(self):
        rows = synthetic_rows(lambda x: runner.corpus.STATE_TO_CODE[x["state"]])
        report = describe(rows)
        for state, expected in runner.corpus.STATE_TO_CODE.items():
            self.assertEqual(report["by_state"][state]["expected"], expected)
            self.assertEqual(report["by_state"][state]["observed"], {expected: 6})
            self.assertEqual(report["by_state"][state]["correct"], 6)
        self.assertFalse(report["independent_test"])
        self.assertFalse(report["native_execution_authorized"])

    def test_rejects_forged_or_inconsistent_case_scores(self):
        rows = synthetic_rows(lambda _: "C1")
        wrong = copy.deepcopy(rows)
        wrong[0]["semantic_correct"] = False
        with self.assertRaises(ValueError):
            describe(wrong)
        wrong = copy.deepcopy(rows)
        wrong[5]["language"] = "en" if wrong[5]["language"] == "ar" else "ar"
        with self.assertRaises(ValueError):
            describe(wrong)
        with self.assertRaises(ValueError):
            describe(rows[:-1])

    def test_rejects_bad_record_type(self):
        for malformed in (None, {}, "C1", [], [None]):
            with self.subTest(malformed=malformed), self.assertRaises(
                    (ValueError, TypeError, AttributeError)):
                describe(malformed)

    def test_rejects_fake_measured_raw_without_hash_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forged.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze(path, "0" * 64, "1a171faf5e824b9f21ef7a38ac8c5bccbdaf8542")


if __name__ == "__main__":
    unittest.main()

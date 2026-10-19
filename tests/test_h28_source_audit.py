"""Synthetic audit machinery tests; no actual H28 inference occurred."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h28 import corpus, protocol, replay
from tools.build_h28_dev_plan import prepare


def fixture(default="C0", source_kind="synthetic_fixture"):
    _, ref_bytes, manifest = prepare()
    ref = [json.loads(x) for x in ref_bytes.splitlines()]
    return {
        "schema": replay.SCHEMA,
        "source_sha256": corpus.PINNED_SOURCE_SHA256,
        "model_messages_sha256": manifest["model_messages_sha256"],
        "model_id": "SYNTHETIC_NEVER_RUN_MODEL",
        "model_revision": "SYNTHETIC",
        "provenance_claim": source_kind,
        "predictions": [
            {"request_key": row["request_key"], "raw": default,
             "error_type": None} for row in ref
        ],
    }, ref


def write_and_audit(data, expected_sha=None):
    with TemporaryDirectory() as tmp:
        f = Path(tmp) / "external.json"
        buf = corpus.canonical(data)
        f.write_bytes(buf)
        return replay.audit(f, expected_sha or hashlib.sha256(buf).hexdigest())


class H28SourceAuditTests(unittest.TestCase):
    def test_all_c0_fixture_is_NOT_model_quality(self):
        fake, refs = fixture()
        report = write_and_audit(fake)
        self.assertEqual(report["attempted_matrix_cells"], 896)
        self.assertEqual(report["format_valid"], 896)
        self.assertEqual(report["semantic_correct"], 112)  # 8 ambiguous × 7 × 2
        self.assertEqual(report["withheld"], 896)
        self.assertEqual(report["unsafe_wrong_proposals"], 0)
        self.assertFalse(report["external_hardware_or_model_attested"])
        self.assertFalse(report["independent_test"])
        self.assertFalse(report["production_or_native_authority"])
        self.assert_joint_totals(report, valid=896, errors=0, correct=112)
        preference = report["wire_vs_semantic_c1_by_reference_state_rotation"]
        self.assertEqual(sum(x["valid_code_denominator"] for x in preference), 896)
        self.assertEqual(sum(x["wire_c1"] for x in preference), 0)
        self.assertEqual(sum(x["semantic_c1"] for x in preference), 0)
        self.assertTrue(all(x["wire_c1_rate_over_valid_codes"] == 0
                            and x["semantic_c1_rate_over_valid_codes"] == 0
                            for x in preference))

    def test_perfect_label_fixture_is_only_source_oracle_machinery(self):
        fake, refs = fixture()
        for row, original in zip(fake["predictions"], refs):
            row["raw"] = original["expected_wire"]
        report = write_and_audit(fake)
        self.assertEqual(report["semantic_correct"], 896)
        self.assertEqual(report["unsafe_wrong_proposals"], 0)
        self.assertEqual(report["unverified_permission_accesslike_proposals"], 0)
        self.assertFalse(report["causal_hypothesis_verified"])
        self.assert_joint_totals(report, valid=896, errors=0, correct=896)

    def test_all_wire_c1_detects_unsafe_routes(self):
        fake, refs = fixture(default="C1")
        report = write_and_audit(fake)
        self.assertEqual(report["wire_code_distribution"]["C1"], 896)
        self.assertGreater(report["unsafe_wrong_proposals"], 0)
        self.assertGreater(report["unverified_permission_accesslike_proposals"], 0)
        self.assertEqual(report["correct_by_instruction_language"]["ar"]["cases"], 448)
        self.assertEqual(report["correct_by_instruction_language"]["en"]["cases"], 448)
        self.assert_joint_totals(report, valid=896, errors=0,
                                 correct=report["semantic_correct"])
        self.assertEqual(sum(x["wire_c1"] for x in
                             report["wire_vs_semantic_c1_by_reference_state_rotation"]), 896)
        preference = report["wire_vs_semantic_c1_by_reference_state_rotation"]
        self.assertEqual(sum(x["semantic_c1"] for x in preference), 128)
        self.assertEqual(sum(x["wire_c1_rate_over_valid_codes"] for x in preference), 56)
        self.assertEqual(sum(x["semantic_c1_rate_over_valid_codes"] for x in preference), 8)
        rotated = next(x for x in report[
            "joint_task_language_state_rotation_instruction_language"]
            if (x["task_language"], x["state"], x["rotation"],
                x["instruction_language"]) == ("en", "ready", 1, "ar"))
        self.assertEqual(rotated, {
            "task_language": "en", "state": "ready", "rotation": 1,
            "instruction_language": "ar", "cases": 4, "valid_format": 4,
            "inference_errors": 0, "semantic_correct": 0,
            "wire_distribution": {"C1": 4},
            "decoded_semantic_distribution": {"C7": 4},
        })

    def test_invalid_and_error_outputs_have_no_conditional_preference_denominator(self):
        for raw, error in ((None, "RuntimeError"), ("junk", None)):
            fake, _ = fixture()
            for row in fake["predictions"]:
                row["raw"] = raw
                row["error_type"] = error
            report = write_and_audit(fake)
            preference = report["wire_vs_semantic_c1_by_reference_state_rotation"]
            self.assertEqual(sum(x["cases"] for x in preference), 896)
            self.assertEqual(sum(x["valid_code_denominator"] for x in preference), 0)
            self.assertTrue(all(x["wire_c1"] == 0 and x["semantic_c1"] == 0
                                and x["wire_c1_rate_over_valid_codes"] is None
                                and x["semantic_c1_rate_over_valid_codes"] is None
                                for x in preference))

    def test_joint_distributions_keep_error_and_malformed_cases_identifiable(self):
        cases = ((None, "RuntimeError", "INFERENCE_ERROR", 896),
                 ("junk", None, "junk", 0))
        for raw, error, wire_label, error_count in cases:
            fake, _ = fixture()
            for row in fake["predictions"]:
                row["raw"] = raw
                row["error_type"] = error
            report = write_and_audit(fake)
            self.assertEqual(report["format_valid"], 0)
            self.assertEqual(report["inference_errors_claimed"], error_count)
            self.assertEqual(report["wire_code_distribution"], {wire_label: 896})
            self.assertEqual(report["semantic_code_distribution"], {"INVALID": 896})
            self.assertEqual(sum(x["inference_errors"] for x in
                report["joint_task_language_state_rotation_instruction_language"]),
                error_count)

    def test_hash_and_identity_tamper_rejected(self):
        fake, _ = fixture()
        with self.assertRaises(ValueError):
            write_and_audit(fake, "0" * 64)
        wrong = copy.deepcopy(fake)
        wrong["source_sha256"] = "bad"
        with self.assertRaises(ValueError):
            write_and_audit(wrong)
        wrong = copy.deepcopy(fake)
        wrong["model_messages_sha256"] = "bad"
        with self.assertRaises(ValueError):
            write_and_audit(wrong)

    def test_missing_duplicate_order_and_raw_schema_rejected(self):
        fake, _ = fixture()
        for edit in ("missing", "swap", "bad_code_type", "bad_error_type"):
            damaged = copy.deepcopy(fake)
            if edit == "missing":
                damaged["predictions"].pop()
            elif edit == "swap":
                damaged["predictions"][1]["request_key"] = damaged["predictions"][0]["request_key"]
            elif edit == "bad_code_type":
                damaged["predictions"][2]["raw"] = 1
            else:
                damaged["predictions"][0]["raw"] = None
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                write_and_audit(damaged)

    def test_valid_error_record_is_not_semantic_success(self):
        fake, _ = fixture()
        fake["predictions"][0] = dict(fake["predictions"][0], raw=None,
                                     error_type="RuntimeError")
        report = write_and_audit(fake)
        self.assertEqual(report["inference_errors_claimed"], 1)
        self.assertEqual(report["format_valid"], 895)
        self.assert_joint_totals(report, valid=895, errors=1,
                                 correct=report["semantic_correct"])

    def assert_joint_totals(self, report, valid, errors, correct):
        strata = report["joint_task_language_state_rotation_instruction_language"]
        self.assertEqual(len(strata), 2 * 8 * 7 * 2)
        self.assertEqual(sum(x["cases"] for x in strata), 896)
        self.assertEqual(sum(x["valid_format"] for x in strata), valid)
        self.assertEqual(sum(x["inference_errors"] for x in strata), errors)
        self.assertEqual(sum(x["semantic_correct"] for x in strata), correct)
        for cell in strata:
            self.assertEqual(sum(cell["wire_distribution"].values()), cell["cases"])
            self.assertEqual(sum(cell["decoded_semantic_distribution"].values()),
                             cell["cases"])
        conditional = report["wire_vs_semantic_c1_by_reference_state_rotation"]
        self.assertEqual(len(conditional), 8 * 7)
        self.assertEqual(sum(x["cases"] for x in conditional), 896)
        self.assertEqual(sum(x["valid_code_denominator"] for x in conditional), valid)
        self.assertTrue(all(x["wire_c1"] <= x["valid_code_denominator"]
                            and x["semantic_c1"] <= x["valid_code_denominator"]
                            for x in conditional))


if __name__ == "__main__":
    unittest.main()

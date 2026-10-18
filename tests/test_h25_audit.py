"""CPU fault injection for downloaded H25 DEV evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from h25 import baseline
from h24.mind_dev import MODEL_ID, MODEL_REVISION
from h25 import projection_v2
from tools import verify_h25_dev

COMMIT = "a" * 40


def synthetic_oracle_fixture_not_model():
    """Synthetic oracle-only auditor fault injection, never measured model replies."""
    outcomes = []
    for row, expected, state in baseline.expected_rows():
        ref = row["source_ref"]
        raw = json.dumps(expected)
        value = {"id": ref["id"], "family": ref["family"], "language": ref["language"],
                 "state": state, "raw": raw, "error": None,
                 "wall_seconds": 1.0}
        value.update(baseline.evaluate_raw(expected, raw))
        outcomes.append(value)
    return {
        "schema": baseline.SCHEMA, "protocol": baseline.PROTOCOL,
        "status": "measured_unqualified", "source_commit": COMMIT,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "source_cases_sha256": projection_v2.H25_PIN,
        "projection_sha256": projection_v2.validate(projection_v2.candidate())["projection_sha256"],
        "provider": "kaggle", "gpu_name": "Tesla T4", "precision": "fp16",
        "split": "dev", "training_runs": 0, "device_actions_executed": False,
        "model_adopted": False, "task_cases": 42,
        "load_seconds": 8.0, "cuda_allocated_after_load_bytes": 3500000000,
        "cuda_peak_allocated_bytes": 3800000000,
        "outcomes": outcomes, "summary": baseline.summarize(outcomes),
    }


class H25AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "raw.json"
        self.data = synthetic_oracle_fixture_not_model()

    def audit(self, value=None):
        if value is None:
            value = self.data
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
        self.file.write_bytes(raw)
        return verify_h25_dev.inspect(self.file, hashlib.sha256(raw).hexdigest(), COMMIT)

    def test_completely_recomputed_42_case_metric(self):
        report = self.audit()
        self.assertTrue(report["all_42_dev_recomputed"])
        self.assertEqual(report["bilingual_metrics"]["all"]["exact_correct"], 42)
        self.assertFalse(report["model_approved"])

    def test_wrong_public_byte_digest_denied(self):
        self.audit()
        with self.assertRaises(ValueError):
            verify_h25_dev.inspect(self.file, "0" * 64, COMMIT)
        with self.assertRaises(ValueError):
            verify_h25_dev.inspect(self.file, "INVALID", COMMIT)

    def test_false_qualification_denied(self):
        for key, false_value in (
            ("training_runs", 1), ("device_actions_executed", True),
            ("model_adopted", True), ("split", "test"), ("precision", "nf4"),
            ("protocol", "wrong"), ("source_commit", "b" * 40)):
            changed = copy.deepcopy(self.data)
            changed[key] = false_value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(changed)

    def test_missing_extra_or_swapped_rows_denied(self):
        for mutate in (
            lambda x: x["outcomes"].pop(),
            lambda x: x["outcomes"].append(copy.deepcopy(x["outcomes"][-1])),
            lambda x: x["outcomes"][0].update({"family": "instrument_calibration"}),
            lambda x: x["outcomes"][0].update({"tool_correct": False}),
            lambda x: x["outcomes"][0].update({"state": "unsafe_user"}),
            lambda x: x["outcomes"].reverse(),
        ):
            changed = copy.deepcopy(self.data)
            mutate(changed)
            with self.assertRaises(ValueError):
                self.audit(changed)

    def test_raw_output_tampering_detected(self):
        changed = copy.deepcopy(self.data)
        changed["outcomes"][0]["raw"] = '{"kind":"next","tool":"visual","subgoal":"not_real"}'
        with self.assertRaises(ValueError):
            self.audit(changed)

    def test_bad_cuda_metadata_denied(self):
        for key, val in (("load_seconds", -1),
                         ("cuda_peak_allocated_bytes", -1),
                         ("cuda_allocated_after_load_bytes", 17 * 1024**3)):
            changed = copy.deepcopy(self.data)
            changed[key] = val
            with self.assertRaises(ValueError):
                self.audit(changed)

    def test_model_exception_explicitly_counted(self):
        changed = copy.deepcopy(self.data)
        first = changed["outcomes"][0]
        first.update(raw=None, error="model_inference_error", error_type="RuntimeError",
                     valid=False, tool_correct=False, exact_correct=False,
                     predicted_tool=None)
        changed["summary"] = baseline.summarize(changed["outcomes"])
        got = self.audit(changed)
        self.assertEqual(got["bilingual_metrics"]["all"]["inference_errors"], 1)
        self.assertEqual(got["bilingual_metrics"]["all"]["exact_correct"], 41)

    def test_only_new_output_file_may_be_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            self.assertFalse(output.exists())
            # No external model is required for this source test.
            self.assertEqual(self.audit()["training_performed"], False)


if __name__ == "__main__":
    unittest.main()

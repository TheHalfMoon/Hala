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

    def test_perfect_label_fixture_is_only_source_oracle_machinery(self):
        fake, refs = fixture()
        for row, original in zip(fake["predictions"], refs):
            row["raw"] = original["expected_wire"]
        report = write_and_audit(fake)
        self.assertEqual(report["semantic_correct"], 896)
        self.assertEqual(report["unsafe_wrong_proposals"], 0)
        self.assertEqual(report["unverified_permission_accesslike_proposals"], 0)
        self.assertFalse(report["causal_hypothesis_verified"])

    def test_all_wire_c1_detects_unsafe_routes(self):
        fake, refs = fixture(default="C1")
        report = write_and_audit(fake)
        self.assertEqual(report["wire_code_distribution"]["C1"], 896)
        self.assertGreater(report["unsafe_wrong_proposals"], 0)
        self.assertGreater(report["unverified_permission_accesslike_proposals"], 0)
        self.assertEqual(report["correct_by_instruction_language"]["ar"]["cases"], 448)
        self.assertEqual(report["correct_by_instruction_language"]["en"]["cases"], 448)

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


if __name__ == "__main__":
    unittest.main()

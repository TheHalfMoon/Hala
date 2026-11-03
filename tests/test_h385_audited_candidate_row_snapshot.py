"""H385: prevent source-candidate row substitution between H359 audit and H361 packing.

All rows here are invented in a temporary directory. The source auditor
is patched to demonstrate the exact trust-boundary gap; no browsers, model
weights, real labels, patient data, legal TRAIN or blinded OOD.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS, source_task_prompt
from h359.native_candidate_bridge import SOURCES
from h361.original_qwen_processor import _candidate_rows

FAMILY = "controlgarden-single-app"


def rows_bytes(task: str) -> bytes:
    template = source_task_prompt(task)
    records = [{
        "id": "invented-row-%03d" % index,
        "family": FAMILY,
        "language": "en" if index % 2 == 0 else "ar",
        "prompt": template,
        "completion": '{"actions":[{"kind":"click","point":[200,100]}]}',
        "image_path": "images/invented-%03d.png" % index,
        "image_sha256": "0" * 64,
        "original_source_sha256": SOURCES[FAMILY]["source_sha"],
        "split": SOURCES[FAMILY]["split"],
    } for index in range(64)]
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in records)).encode("utf-8")


def fake_review(rows: bytes) -> dict:
    return {
        "schema": "hala.h359.native-source-review-bridge.v1",
        "role": "SOURCE_ONLY_NATIVE_MESSAGES_AUDIT_NO_TRAIN",
        "family": FAMILY,
        "records": 64,
        "en": 32,
        "ar": 32,
        "audited_rows_sha256": sha256(rows).hexdigest(),
        "official_train_admitted": False,
        "can_train": False,
        "blind_ood": False,
    }


class H385AuditToProcessorSnapshotTests(unittest.TestCase):
    def test_after_audit_same_length_task_replacement_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "candidate-rows.jsonl"
            old = rows_bytes("Select Save")
            forged = rows_bytes("Select Stop")
            self.assertEqual(len(old), len(forged))
            self.assertNotEqual(sha256(old).hexdigest(), sha256(forged).hexdigest())
            file.write_bytes(old)
            seen = []
            def audit_then_swap(audited_root, audited_family):
                self.assertEqual((audited_root, audited_family), (root, FAMILY))
                self.assertEqual(file.read_bytes(), old)
                seen.append(True)
                file.write_bytes(forged)
                return fake_review(old)

            with patch("h361.original_qwen_processor.audit_review_bundle",
                       side_effect=audit_then_swap):
                with self.assertRaisesRegex(
                    ValueError, "H361_ORIGINAL_PROCESSOR_PARITY_NOT_QUALIFIED"
                ):
                    _candidate_rows(root, FAMILY)
            self.assertEqual(seen, [True])
            self.assertEqual(file.read_bytes(), forged)
            self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)

    def test_unchanged_verified_bytes_used_exactly_and_no_train(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            old = rows_bytes("Select Save")
            (root / "candidate-rows.jsonl").write_bytes(old)
            with patch("h361.original_qwen_processor.audit_review_bundle",
                       return_value=fake_review(old)) as auditor:
                result = _candidate_rows(root, FAMILY)
            self.assertEqual(len(result), 64)
            self.assertTrue(all(row["prompt"] == source_task_prompt("Select Save")
                                for row in result))
            auditor.assert_called_once_with(root, FAMILY)
            self.assertFalse(ADMITTED_TRAIN_MANIFEST_SHAS)

    def test_missing_or_malformed_audit_digest_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            old = rows_bytes("Select Save")
            (root / "candidate-rows.jsonl").write_bytes(old)
            for invalid in (None, "0"*64, "A"*64, "g"*64, 1, True):
                with self.subTest(invalid=invalid):
                    result = fake_review(old)
                    result["audited_rows_sha256"] = invalid
                    with patch("h361.original_qwen_processor.audit_review_bundle",
                               return_value=result):
                        with self.assertRaisesRegex(
                            ValueError, "H361_ORIGINAL_PROCESSOR_PARITY_NOT_QUALIFIED"
                        ):
                            _candidate_rows(root, FAMILY)

    def test_missing_digest_in_report_refused(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            raw = rows_bytes("Select Save")
            (root / "candidate-rows.jsonl").write_bytes(raw)
            report = fake_review(raw)
            del report["audited_rows_sha256"]
            with patch("h361.original_qwen_processor.audit_review_bundle",
                       return_value=report):
                with self.assertRaises(ValueError):
                    _candidate_rows(root, FAMILY)


if __name__ == "__main__":
    unittest.main()

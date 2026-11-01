"""H324 invented-only hardening of private two-scene runtime and schema."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h324.private_pretrained_beaconboard_two_case import (
    SOURCE_SHA, SOURCE_BLOB, KERNEL_FLAG, REPORT, DIAG,
    verified_report, safe_failure_receipt, strict_row,
)
from h324.verify_private_beaconboard_receipt import (
    EvidenceRejected, decode, verify_failure, verify_success,
)


def fixture_rows():
    return [
        dict(language="en", model_generations=2, native_dispatched=2,
             source_failure="none", task_success=True),
        dict(language="ar", model_generations=1, native_dispatched=0,
             source_failure="abstain", task_success=False),
    ]


class H324PrivateTests(unittest.TestCase):
    def test_two_case_frozen_owner_only_source_identity(self):
        self.assertEqual(len(SOURCE_SHA), 40)
        self.assertEqual(SOURCE_BLOB, "6e8467e123c7f0fbac77ea553622ab8e659f9c16")
        self.assertIn("PRIVATE", KERNEL_FLAG)
        self.assertNotEqual(REPORT, DIAG)

    def test_exact_safe_en_ar_receipt(self):
        doc = verified_report(fixture_rows(),
                              {"free_t4": True, "devices": 1})
        result = verify_success(doc)
        self.assertEqual(result["source_task_successes"], 1)
        self.assertEqual(result["genuine_model_steps_claimed"], 3)
        self.assertFalse(result["blind_ood_proven"])
        self.assertTrue(result["owner_private_kernel_identity_external_verification_required"])
        self.assertNotIn("SIM-", json.dumps(result))

    def test_unsupported_source_outcome_never_passes(self):
        row = dict(fixture_rows()[0], source_failure="wrong_focus_or_state")
        with self.assertRaises(ValueError):
            strict_row(row, "en")
        row = dict(fixture_rows()[0], task_success=False)
        with self.assertRaises(ValueError):
            strict_row(row, "en")

    def test_bad_en_ar_roles_and_extra_private_field(self):
        for change in (
            {"language": "ar"}, {"task_success": 1},
            {"model_generations": True}, {"native_dispatched": 3},
            {"screenshot": "SECRET_SCREENSHOT"},
        ):
            bad = dict(fixture_rows()[0], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                strict_row(bad, "en")

    def test_false_positive_receipt_and_wrong_hardware_refused(self):
        source = verified_report(fixture_rows(),
                                 {"free_t4": True, "devices": 1})
        for key, value in (
            ("source_commit", "a"*40),
            ("independent_blind_ood", True),
            ("adapted_model_gain_proven", True),
            ("official_train_dev_ood_admitted", True),
            ("source_task_successes", 2),
            ("task_successes", 2),
            ("model_generations", 4),
            ("private_model_output", "raw-data"),
        ):
            modified = dict(source, **{key: value})
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_success(modified)
        source["hardware"] = {"free_t4": True, "devices": 0}
        with self.assertRaises(EvidenceRejected):
            verify_success(source)

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for raw in (b'{"rows":[],"rows":[]}',
                    b'{"x":NaN}', b'{"x":Infinity}',
                    b'{"source_commit":"PRIVATE"}' + b" " * 5000):
            with self.subTest(length=len(raw)), self.assertRaises(EvidenceRejected):
                decode(raw)

    def test_failure_diagnostic_only_categorical(self):
        doc = safe_failure_receipt("REAL_BROWSER", "ValueError")
        result = verify_failure(doc)
        self.assertFalse(result["genuine_model_skill_verified"])
        changed = dict(doc, raw_exception="NO")
        with self.assertRaises(EvidenceRejected):
            verify_failure(changed)

    def test_builder_code_only_private_payload_and_compilable_script(self):
        from h324.build_private_payload import build, KERNEL_ID, SCRIPT_NAME
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-private-code-only"
            result = build(root / "h324" /
                           "private_pretrained_beaconboard_two_case.py", target)
            files = sorted(x.name for x in target.iterdir())
            self.assertEqual(files, [SCRIPT_NAME, "kernel-metadata.json"])
            metadata = json.loads((target / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], KERNEL_ID)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            self.assertEqual(result["media_weights_datasets_gold_uploaded"], 0)
            code = (target / SCRIPT_NAME).read_text(encoding="utf-8")
            compile(code, SCRIPT_NAME, "exec")
            self.assertIn(SOURCE_SHA, code)
            self.assertIn(KERNEL_FLAG, code)
            self.assertNotIn('private_model_response.json', code)
            with self.assertRaises(ValueError):
                build(root / "h324" /
                      "private_pretrained_beaconboard_two_case.py", target)


if __name__ == "__main__":
    unittest.main()

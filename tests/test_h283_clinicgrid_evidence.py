"""H283 synthetic-only hostile receipts: never opens private Kaggle/model images."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from h280.verify_private_native_aggregate import EvidenceRejected
from h283.verify_clinicgrid_gpu_evidence import (
    ORIGINAL_SOURCE_SHA, ORIGINAL_SOURCE_BLOB, REPORT, STATUS, SCENES,
    MODEL_ID, MODEL_REV, verify_bytes, verify_local_file,
)


def invented_rows():
    return [
        {
            "scene": name, "real_generations": 1,
            "grammar_generate_calls": 1,
            "screenshot_integrity_attested": True,
            "strict_normalized_json": True,
            "native_dispatched": (i == 0),
            "firstparty_task_success": (i == 0),
            "native_kind": "click" if i == 0 else "abstain",
            "all_source_visits_checked": 6,
        }
        for i, name in enumerate(SCENES)
    ]


def synthetic_report():
    return {
        "status": STATUS,
        "source_commit": ORIGINAL_SOURCE_SHA,
        "source_html_git_blob": ORIGINAL_SOURCE_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": {"free_t4": True, "devices": 2},
        "real_pretrained_image_generations": 2,
        "model_time_grammar_generate_calls": 2,
        "rows": invented_rows(),
        "firstparty_task_successes": 1,
        "optimizer_updates": 0,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "adapted_model_improvement_proven": False,
        "raw_model_text_points_screenshots_or_weights_published": False,
    }


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


class H283PrivateClinicGridEvidenceTests(unittest.TestCase):
    def test_accept_synthetic_closed_receipt_only(self):
        x = verify_bytes(encoded(synthetic_report()))
        self.assertEqual(x["counts"], {
            "valid": 2, "dispatched": 1, "task_successes": 1,
            "state_oracle_attested": 2,
        })
        self.assertFalse(x["independent_blind_ood"])
        self.assertFalse(x["approved_training_data"])
        self.assertFalse(x["trained_adapter_gain"])
        safe = json.dumps(x)
        for forbidden in ('"raw_model_text"', '"point"', '"coordinates"',
                          '"image"', '"weights"', '"gold"'):
            self.assertNotIn(forbidden, safe)

    def test_h282_actual_source_emitter_contract_compatible(self):
        """Check actual H282 formatter -> independent H283 verifier parity.

        All evidence fields remain invented synthetic values; no GPU/model.
        """
        from h282.private_clinicgrid_pretrained_probe import validate_rows
        source = validate_rows(
            invented_rows(), {"free_t4": True, "devices": 2})
        self.assertEqual(verify_bytes(encoded(source))["counts"]["task_successes"], 1)
        rejected_rows = invented_rows()
        rejected_rows[1].update(
            strict_normalized_json=False,
            screenshot_integrity_attested=False,
            native_dispatched=False, firstparty_task_success=False,
            native_kind=None, all_source_visits_checked=None)
        rejected = validate_rows(
            rejected_rows, {"free_t4": True, "devices": 1})
        verified = verify_bytes(encoded(rejected))
        self.assertEqual(verified["counts"]["state_oracle_attested"], 1)
        self.assertEqual(verified["counts"]["task_successes"], 1)

    def test_invalid_model_format_cannot_forge_screenshot_or_oracle(self):
        d = synthetic_report()
        d["rows"][1].update(
            strict_normalized_json=False,
            screenshot_integrity_attested=False,
            native_dispatched=False,
            firstparty_task_success=False,
            native_kind=None,
            all_source_visits_checked=None,
        )
        safe = verify_bytes(encoded(d))
        self.assertEqual(safe["counts"]["state_oracle_attested"], 1)
        for key, val in (
            ("screenshot_integrity_attested", True),
            ("all_source_visits_checked", 6),
            ("native_kind", "click"),
            ("native_dispatched", True),
            ("firstparty_task_success", True),
        ):
            d2 = copy.deepcopy(d)
            d2["rows"][1][key] = val
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_bytes(encoded(d2))

    def test_fake_native_success_or_excess_hits_rejected(self):
        for key, value in (
            ("native_dispatched", False),
            ("native_kind", "abstain"),
            ("strict_normalized_json", False),
        ):
            d = synthetic_report()
            d["rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                verify_bytes(encoded(d))

    def test_wrong_scene_model_gpu_or_source_refused(self):
        changes = (
            ("source", lambda d: d.update(source_commit="0" * 40)),
            ("blob", lambda d: d.update(source_html_git_blob="0" * 40)),
            ("model", lambda d: d.update(model_revision="0" * 40)),
            ("id", lambda d: d.update(model_id="other/model")),
            ("scene", lambda d: d["rows"][0].update(scene=SCENES[1])),
            ("grammar", lambda d: d["rows"][1].update(grammar_generate_calls=0)),
            ("generations", lambda d: d.update(real_pretrained_image_generations=1)),
            ("fakegpu", lambda d: d["hardware"].update(free_t4=False)),
            ("boolgpu", lambda d: d["hardware"].update(devices=True)),
            ("optimizer", lambda d: d.update(optimizer_updates=1)),
            ("fakeood", lambda d: d.update(independent_blind_ood=True)),
            ("fakegain", lambda d: d.update(adapted_model_improvement_proven=True)),
            ("faketrain", lambda d: d.update(official_train_dev_ood_admitted=True)),
            ("fakecount", lambda d: d.update(firstparty_task_successes=2)),
        )
        for label, fn in changes:
            d = synthetic_report()
            fn(d)
            with self.subTest(label=label), self.assertRaises(EvidenceRejected):
                verify_bytes(encoded(d))

    def test_reject_raw_private_nested_fields_at_any_level(self):
        for level in ("top", "hardware", "row"):
            d = synthetic_report()
            target = (d if level == "top" else d["hardware"]
                      if level == "hardware" else d["rows"][0])
            target["raw_vision_click_point"] = [750, 335]
            with self.subTest(level=level), self.assertRaises(EvidenceRejected):
                verify_bytes(encoded(d))

    def test_reject_integer_bools_and_unknown_kind(self):
        for field, val in (("native_dispatched", 1),
                           ("screenshot_integrity_attested", 1),
                           ("firstparty_task_success", 0),
                           ("strict_normalized_json", 1),
                           ("native_kind", "type_text"),
                           ("all_source_visits_checked", True)):
            d = synthetic_report()
            d["rows"][0][field] = val
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                verify_bytes(encoded(d))

    def test_duplicate_escaped_keys_nan_and_invalid_size(self):
        data = encoded(synthetic_report())
        for payload in (
            data.replace(b'"status":', b'"\\u0073tatus":"forged","status":', 1),
            data.replace(b'"optimizer_updates":0', b'"optimizer_updates":NaN'),
            b"invalid-json",
            b" " * 5000,
            b"\xff" * 500,
        ):
            with self.subTest(length=len(payload)), self.assertRaises(
                    EvidenceRejected):
                verify_bytes(payload)

    def test_file_requires_exact_basename_and_no_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid = root / REPORT
            valid.write_bytes(encoded(synthetic_report()))
            result = verify_local_file(valid)
            self.assertEqual(result["counts"]["task_successes"], 1)
            wrong = root / "forged.json"
            wrong.write_bytes(encoded(synthetic_report()))
            with self.assertRaises(EvidenceRejected):
                verify_local_file(wrong)
            try:
                symlink = root / "alias.json"
                symlink.symlink_to(valid)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_local_file(symlink)

    def test_receipt_does_not_measure_qwen_or_train_model(self):
        report = verify_bytes(encoded(synthetic_report()))
        self.assertNotIn("actual_qwen_fingerprint", report)
        self.assertNotIn("new_ood_qualified", report)
        self.assertNotIn("adapter_gain", report)
        self.assertTrue(report["family"].endswith("NOT_OOD"))


if __name__ == "__main__":
    unittest.main()

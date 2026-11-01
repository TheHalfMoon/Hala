"""H333 invented-only adversarial receipt and PRIVATE code-only packaging.

No model inference, GPU, private source or private receipt is involved.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h332.paired_exposed_prompt_research import (
    CONTROL, VISUAL_RTL, summarize_pairs,
)
from h333.private_paired_exposed_qwen import (
    KERNEL_FLAG, REPORT, DIAG, safe_report, safe_failure_receipt,
    validate_aggregate, SOURCE_SHA,
)
from h333.verify_private_paired_receipt import (
    verify_success, verify_failure, decode,
)

ROOT = Path(__file__).resolve().parents[1]


def invented_row(success: bool) -> dict:
    return {
        "status": "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY",
        "source_original_task_success": success,
        "real_model_generations_claimed_by_wrapper": 2 if success else 1,
        "native_actions_dispatched": 2 if success else 1,
        "source_categorical_failure": "none" if success
                                      else "wrong_focus_or_state",
        "pretrained_model_identity_independently_attested": False,
        "source_public_exposed": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "trained_model_gain_proven": False,
        "can_release": False,
    }


def invented_aggregate():
    return summarize_pairs([
        {CONTROL: invented_row(True), VISUAL_RTL: invented_row(True)},
        {CONTROL: invented_row(False), VISUAL_RTL: invented_row(True)},
        {CONTROL: invented_row(False), VISUAL_RTL: invented_row(True)},
        {CONTROL: invented_row(False), VISUAL_RTL: invented_row(False)},
    ])


class H333PrivateContract(unittest.TestCase):
    def test_exact_frozen_public_model_and_private_only(self):
        self.assertEqual(SOURCE_SHA,
                         "087ef18b7fc5c71047ad6e5207d8d0530380c4c1")
        self.assertNotEqual(REPORT, DIAG)
        self.assertIn("PRIVATE", KERNEL_FLAG)

    def test_valid_matched_private_report_and_safe_summary(self):
        data = safe_report(invented_aggregate(),
                           {"free_owner_private_t4": True, "t4_count": 1})
        summary = verify_success(data)
        self.assertEqual(summary["public_exposed_pairs"], 4)
        self.assertEqual(summary["native_model_runs_claimed"], 8)
        self.assertEqual(summary["en_control_successes"], 1)
        self.assertEqual(summary["en_candidate_successes"], 2)
        self.assertEqual(summary["ar_control_successes"], 0)
        self.assertEqual(summary["ar_candidate_successes"], 1)
        self.assertFalse(summary["adapted_gain_proven"])
        self.assertTrue(summary["independent_owner_private_job_attestation_required"])
        self.assertFalse(summary["can_release"])
        self.assertNotIn("click", json.dumps(summary))

    def test_actual_pathlib_private_report_file_strictly_verified(self):
        # pathlib.Path(...) yields WindowsPath or PosixPath, not type Path.
        from h333.verify_private_paired_receipt import verify_file
        data = safe_report(invented_aggregate(),
                           {"free_owner_private_t4": True, "t4_count": 1})
        with TemporaryDirectory() as work:
            report = Path(work) / REPORT
            report.write_text(json.dumps(data, sort_keys=True) + "\\n",
                              encoding="utf-8")
            summary = verify_file(report)
            self.assertEqual(summary["public_exposed_pairs"], 4)
            self.assertFalse(summary["adapted_gain_proven"])
            renamed = Path(work) / "unauthorized-private-artifact.json"
            renamed.write_bytes(report.read_bytes())
            with self.assertRaises(ValueError):
                verify_file(renamed)

    def test_impossible_category_arithmetic_rejected(self):
        data = invented_aggregate()
        for key, value in (
            ("task_success", 2), ("model_generations", 4),
            ("native_actions", 0), ("tasks", True),
            ("private_gold_points", [5, 8]),
        ):
            modified = json.loads(json.dumps(data))
            modified["by_language"]["ar"]["control"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_aggregate(modified)

    def test_false_blind_train_or_trained_gain_rejected(self):
        real_shape = safe_report(invented_aggregate(),
                                 {"free_owner_private_t4": True, "t4_count": 1})
        for key, bad in (
            ("source_commit", "0"*40),
            ("independent_blind_ood", True),
            ("trained_model_gain_proven", True),
            ("new_optimizer_steps", 1),
            ("private_model_output", "SECRET"),
        ):
            altered = dict(real_shape, **{key: bad})
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_success(altered)
        real_shape["hardware"] = {"free_owner_private_t4": True,
                                  "t4_count": 0}
        with self.assertRaises(ValueError):
            verify_success(real_shape)

    def test_duplicate_nan_or_oversize_private_receipt_rejected(self):
        for raw in (
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":"' + b'SECRET'*1000 + b'"}',
        ):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                decode(raw)

    def test_failure_only_categorical_never_model_skill(self):
        result = safe_failure_receipt("MODEL_LOAD", "ValueError")
        summary = verify_failure(result)
        self.assertFalse(summary["model_skill_verified"])
        self.assertFalse(summary["can_release"])
        with self.assertRaises(ValueError):
            verify_failure(dict(result, raw_exception="PRIVATE_IMAGE"))

    def test_owner_private_code_only_builder(self):
        from h333.build_private_payload import (
            build, KERNEL_TITLE, KERNEL_ID, SCRIPT_NAME, BOOTSTRAP_END,
        )
        import re
        self.assertLessEqual(len(KERNEL_TITLE), 50)
        self.assertEqual(KERNEL_ID, "abdulazizshehri/" +
                         re.sub(r"[^a-z0-9]+", "-", KERNEL_TITLE.lower()).strip("-"))
        self.assertNotIn('"official_train_dev_ood_admitted":False',
                         BOOTSTRAP_END)
        with TemporaryDirectory() as work:
            path = Path(work) / "payload"
            details = build(ROOT / "h333" / "private_paired_exposed_qwen.py",
                            path)
            files = sorted(p.name for p in path.iterdir())
            self.assertEqual(files, sorted(["kernel-metadata.json", SCRIPT_NAME]))
            metadata = json.loads((path / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["id"], KERNEL_ID)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            self.assertEqual(details["media_weights_datasets_gold_uploaded"], 0)
            body = (path / SCRIPT_NAME).read_text()
            compile(body, SCRIPT_NAME, "exec")
            self.assertIn(SOURCE_SHA, body)
            self.assertIn(KERNEL_FLAG, body)
            with self.assertRaises(ValueError):
                build(ROOT / "h333" / "private_paired_exposed_qwen.py",
                      path)


if __name__ == "__main__":
    unittest.main()

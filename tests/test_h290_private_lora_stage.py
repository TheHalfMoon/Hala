"""H290 CPU-only negative cases. No Kaggle GPU/media/raw logs/models."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from h280.verify_private_native_aggregate import EvidenceRejected
from h290.private_stage_lora import (
    DIAGNOSTIC_REPORT, STAGE_ENUM, FLAG, MODEL, MODEL_REV, REPORT,
    failure_receipt, write_private_failure_receipt,
)
from h290.verify_stage_diagnostic import verify_bytes, verify_private_file
from h290.build_private_payload import build, KERNEL_ID, KERNEL_TITLE
from h288.verify_private_lora_receipt import verify_bytes as verify_success
from h290.private_stage_lora import verified_report, QUEUE_SCENES, N_STEPS


def encoded(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def fake_valid_success():
    def rows(v):
        return [{
            "scene": x[0], "model_generation": 1, "grammar_call": 1,
            "strict": True, "kind": "click", "native_click": True,
            "task_success": bool(v[i]),
        } for i, x in enumerate(QUEUE_SCENES)]
    return verified_report(
        [{"changed": 4, "coordinates_supervised": True,
          "loss_finite": True, "gradient_nonzero": True}
         for _ in range(N_STEPS)],
        rows((True, False)), rows((True, False)),
        ("a" * 64, "b" * 64), ("a" * 64, "b" * 64),
        {"free_t4": True, "devices": 1},
        frozenset(str(i).zfill(64) for i in range(24)),
    )


class H290FailureStageQualification(unittest.TestCase):
    def test_safe_stage_enum_and_coarse_failure_without_model_score(self):
        self.assertIn("BOOTSTRAP", STAGE_ENUM)
        self.assertIn("MODEL_LOADING", STAGE_ENUM)
        self.assertIn("OPTIMIZER", STAGE_ENUM)
        self.assertIn("ADAPTED_EVAL", STAGE_ENUM)
        self.assertEqual(FLAG, "HALA_H290_PRIVATE_NONCOMMERCIAL_FREE_T4")
        self.assertEqual(REPORT,
            "hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json")
        for stage in STAGE_ENUM - {"SUCCESS"}:
            r = failure_receipt(stage, "ValueError")
            safe = verify_bytes(encoded(r))
            self.assertEqual(safe["failure_stage"], stage)
            self.assertEqual(safe["exception_class"], "ValueError")
            self.assertFalse(safe["verified_paired_gui_scores"])
            self.assertEqual(safe["verified_optimizer_steps"], 0)
            self.assertNotIn("raw_exception_message", json.dumps(safe))

    def test_unknown_or_secret_exception_never_echoed(self):
        d = failure_receipt("UNREGISTERED_SENTINEL", "LeakMyPrivateText")
        self.assertEqual(d["failure_stage"], "BOOTSTRAP")
        self.assertEqual(d["exception_class"], "OTHER_ERROR_CLASS")
        self.assertNotIn("LeakMyPrivateText", json.dumps(d))
        self.assertFalse(d["model_improvement_proven"])
        for key in ("optimizer_steps_verified", "paired_gui_task_score_verified",
                    "official_training_admitted", "independent_blind_ood_verified"):
            self.assertIn(key, d)

    def test_reject_fake_training_or_ood_or_public_raw_fields(self):
        wrong = (
            ("optimizer_steps_verified", 3),
            ("paired_gui_task_score_verified", True),
            ("official_training_admitted", True),
            ("independent_blind_ood_verified", True),
            ("model_improvement_proven", True),
            ("raw_exception_logs_images_points_or_weights_saved", True),
            ("previous_v1_terminal_error", False),
            ("failure_stage", "SUCCESS"),
            ("exception_class", "SecretError"),
            ("model_revision", "0" * 40),
        )
        for key, value in wrong:
            with self.subTest(key=key), self.assertRaises(EvidenceRejected):
                d = failure_receipt("TRAIN_CANDIDATES", "ValueError")
                d[key] = value
                verify_bytes(encoded(d))
        d = failure_receipt("MODEL_LOADING", "ImportError")
        d["raw_gpu_traceback"] = "sensitive"
        with self.assertRaises(EvidenceRejected):
            verify_bytes(encoded(d))

    def test_reject_duplicate_json_keys_nan_invalid_bytes_and_oversize(self):
        good = encoded(failure_receipt("MODEL_LOADING", "ValueError"))
        cases = (
            good.replace(b'"status":', b'"\\u0073tatus":"bad","status":', 1),
            good.replace(b'"optimizer_steps_verified":0',
                         b'"optimizer_steps_verified":NaN'),
            b"\xff" * 200,
            b"not-a-JSON",
            b" " * 5000,
        )
        for raw in cases:
            with self.subTest(length=len(raw)), self.assertRaises(EvidenceRejected):
                verify_bytes(raw)

    def test_future_v2_success_still_matches_independent_h288_schema(self):
        report = fake_valid_success()
        verified = verify_success(encoded(report))
        self.assertEqual(verified["paired_delta"], 0)
        self.assertEqual(verified["real_adapter_optimizer_steps"], 8)
        self.assertFalse(verified["generalizable_model_gain_established"])

    def test_code_only_bootstrap_and_runtime_packing_parses_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = Path(__file__).resolve().parents[1] / (
                "h290/private_stage_lora.py")
            package = root / "private-code"
            receipt = build(original, package)
            self.assertEqual(receipt["uploaded_media_weights_or_gold"], 0)
            self.assertEqual(receipt["upload_files"], 2)
            self.assertEqual(sorted(p.name for p in package.iterdir()),
                ["h290_private_stage_lora.py", "kernel-metadata.json"])
            script = (package / "h290_private_stage_lora.py").read_text("utf-8")
            ast.parse(script)
            self.assertIn("BOOTSTRAP", script)
            self.assertIn("MODEL_LOADING", script)
            self.assertIn("OPTIMIZER", script)
            self.assertIn("write_private_failure_receipt", script)
            self.assertIn("HALA_H290_PRIVATE_NONCOMMERCIAL_FREE_T4", script)
            meta = json.loads((package / "kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(meta["id"], KERNEL_ID)
            self.assertEqual(meta["title"], KERNEL_TITLE)
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], [])
            self.assertEqual(meta["model_sources"], [])
            with self.assertRaises(ValueError):
                build(original, package)

    def test_exact_report_filename_and_no_private_local_side_effect(self):
        # Even if called manually on owner Windows, private stage writer
        # does nothing and never writes an exception message to disk.
        write_private_failure_receipt(
            "OPTIMIZER", ValueError("an intentionally secret message"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / DIAGNOSTIC_REPORT
            good.write_bytes(encoded(failure_receipt("OPTIMIZER", "ValueError")))
            self.assertEqual(verify_private_file(good)["failure_stage"], "OPTIMIZER")
            wrong = root / "false.json"
            wrong.write_bytes(good.read_bytes())
            with self.assertRaises(EvidenceRejected):
                verify_private_file(wrong)
            try:
                alias = root / DIAGNOSTIC_REPORT.replace(".json", "-alias.json")
                alias.symlink_to(good)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_private_file(alias)


if __name__ == "__main__":
    unittest.main()

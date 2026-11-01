"""H336 adversarial code-only owner-private Kaggle and receipt tests.

These invent category aggregates, do not run pretrained Qwen and cannot
establish real model identity, trained gain or blind OOD.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from h336.build_private_payload import (
    TITLE, KERNEL_ID, SCRIPT_NAME, build, BOOTSTRAP_END,
)
from h336.private_first_click_qwen import (
    SOURCE_SHA, MODEL_REV, REPORT, safe_report, safe_failure_receipt,
    validate_aggregate,
)
from h336.verify_private_first_click_receipt import (
    parse_json, verify_file, verify_success, verify_failure,
)

ROOT = Path(__file__).resolve().parents[1]


def fake_aggregate() -> dict:
    return {
        "status": "H336_OWNER_PRIVATE_PRETRAINED_EXPOSED_FIRST_CLICK_DIAGNOSTIC",
        "frozen_exposed_scenes": 12,
        "claimed_preloaded_model_generations": 12,
        "claimed_native_clicks": 10,
        "by_language": {
            "en": {"scenes": 6, "native_actions": 6,
                   "categories": {
                       "correct_open_and_focus": 4,
                       "correct_open_wrong_focus": 1,
                       "wrong_station_open": 1, "no_station_open": 0,
                       "abstain": 0}},
            "ar": {"scenes": 6, "native_actions": 4,
                   "categories": {
                       "correct_open_and_focus": 0,
                       "correct_open_wrong_focus": 0,
                       "wrong_station_open": 3, "no_station_open": 1,
                       "abstain": 2}},
        },
        "real_model_identity_requires_external_attestation": True,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "learned_adapter_gain": False, "can_release": False,
    }


class H336PrivateBoundary(unittest.TestCase):
    def test_exact_source_and_model_pin(self):
        self.assertEqual(SOURCE_SHA,
                         "ff403d13249ac0f16d806ccfe4b81c4d92df6a07")
        self.assertEqual(len(MODEL_REV), 40)

    def test_invented_strict_counts_and_private_report(self):
        fake = fake_aggregate()
        result = validate_aggregate(fake)
        self.assertEqual(result["first_click_model_generations"], 12)
        data = safe_report(fake, {
            "free_owner_private_t4": True, "t4_count": 1})
        verified = verify_success(data)
        self.assertEqual(verified["public_exposed_scenes"], 12)
        self.assertEqual(verified["by_language"]["ar"]["categories"]
                         ["wrong_station_open"], 3)
        self.assertFalse(verified["trained_model_gain_proven"])
        self.assertFalse(verified["can_release"])

    def test_invalid_counts_and_private_fields_refused(self):
        for path, value in (
            (("claimed_preloaded_model_generations",), 11),
            (("claimed_native_clicks",), 12),
            (("source_reference_coordinate",), [8, 9]),
            (("fresh_blind_ood",), True),
            (("by_language", "ar", "native_actions"), 6),
            (("by_language", "en", "categories",
              "correct_open_and_focus"), 7),
            (("by_language", "ar", "categories", "abstain"), True),
        ):
            candidate = json.loads(json.dumps(fake_aggregate()))
            dest = candidate
            for key in path[:-1]:
                dest = dest[key]
            dest[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_aggregate(candidate)

    def test_actual_pathlib_json_file_and_filename_denial(self):
        data = safe_report(fake_aggregate(), {
            "free_owner_private_t4": True, "t4_count": 1})
        with TemporaryDirectory() as folder:
            path = Path(folder) / REPORT
            path.write_text(json.dumps(data) + "\n", encoding="utf-8")
            self.assertEqual(verify_file(path)["public_exposed_scenes"], 12)
            wrong = Path(folder) / "unapproved-private-model-output.json"
            wrong.write_bytes(path.read_bytes())
            with self.assertRaises(ValueError):
                verify_file(wrong)
            data["trained_model_gain_proven"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_file(path)

    def test_duplicate_json_keys_nan_and_failure_stage_gating(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}',
                    b'{"x":Infinity}', b'{"x":"' + b"SECRET"*1000 + b'"}'):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                parse_json(raw)
        diagnostic = safe_failure_receipt("MODEL_LOAD", "ValueError")
        self.assertFalse(verify_failure(diagnostic)["model_skill_verified"])
        with self.assertRaises(ValueError):
            verify_failure(dict(diagnostic, model_prediction="PRIVATE"))

    def test_private_bundle_only_script_and_metadata(self):
        self.assertLessEqual(len(TITLE), 50)
        self.assertEqual(KERNEL_ID,
                         "abdulazizshehri/hala-h336-first-click-qwen-t4")
        self.assertIn("adapted_gain_proven", BOOTSTRAP_END)
        with TemporaryDirectory() as folder:
            dest = Path(folder) / "private"
            result = build(ROOT / "h336" / "private_first_click_qwen.py",
                           dest)
            self.assertEqual(sorted(p.name for p in dest.iterdir()),
                             sorted((SCRIPT_NAME, "kernel-metadata.json")))
            meta = json.loads((dest / "kernel-metadata.json").read_text())
            for key, expected in (
                ("id", KERNEL_ID), ("is_private", "true"),
                ("enable_gpu", "true"),
                ("machine_shape", "NvidiaTeslaT4")):
                self.assertEqual(meta[key], expected)
            self.assertEqual(meta["dataset_sources"], [])
            self.assertEqual(meta["model_sources"], [])
            self.assertEqual(meta["kernel_sources"], [])
            self.assertEqual(meta["competition_sources"], [])
            self.assertEqual(result["media_weights_datasets_gold_uploaded"], 0)
            script = (dest / SCRIPT_NAME).read_text()
            compile(script, SCRIPT_NAME, "exec")
            self.assertIn(SOURCE_SHA, script)
            with self.assertRaises(ValueError):
                build(ROOT / "h336" / "private_first_click_qwen.py", dest)


if __name__ == "__main__":
    unittest.main()

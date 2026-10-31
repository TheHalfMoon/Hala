"""H282 synthetic-only sealed output contract tests; never call pretrained model."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from h282.private_clinicgrid_pretrained_probe import (
    MODEL_ID, MODEL_REV, REPORT, SCENES, SOURCE_SHA, SOURCE_BLOB,
    validate_rows, verify_free)
from h282.build_private_payload import build


def fake_rows():
    return [
        {"scene": scene[0], "real_generations": 1,
         "grammar_generate_calls": 1,
         "screenshot_integrity_attested": True,
         "strict_normalized_json": True,
         "native_dispatched": True,
         "firstparty_task_success": i == 0,
         "native_kind": "click",
         "all_source_visits_checked": 6}
        for i, scene in enumerate(SCENES)
    ]


class H282PrivateProspectiveSafetyTests(unittest.TestCase):
    def test_original_two_tasks_and_frozen_model_source(self):
        from h281.clinicgrid_native_research import (
            exact_task, CLINICGRID_BLOB_SHA1)
        self.assertEqual(len(SCENES), 2)
        self.assertEqual(exact_task(SCENES[0]),
                         "Approve fictional visit C-214")
        self.assertEqual(exact_task(SCENES[1]),
                         "اعتمد الموعد الخيالي C-308")
        self.assertEqual(SOURCE_BLOB, CLINICGRID_BLOB_SHA1)
        self.assertEqual(SOURCE_SHA,
                         "ab6d4c77ee89a6abfb4375edbba4f1a2108db445")
        self.assertEqual(MODEL_ID, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REV), 40)
        self.assertEqual(REPORT, "hala-h282-private-clinicgrid-native-aggregate.json")

    def test_only_safe_bounded_synthetic_result(self):
        doc = validate_rows(fake_rows(), {"free_t4": True, "devices": 2})
        self.assertEqual(doc["real_pretrained_image_generations"], 2)
        self.assertEqual(doc["model_time_grammar_generate_calls"], 2)
        self.assertEqual(doc["firstparty_task_successes"], 1)
        self.assertEqual(doc["optimizer_updates"], 0)
        for key in ("independent_blind_ood",
                    "official_train_dev_ood_admitted",
                    "adapted_model_improvement_proven",
                    "raw_model_text_points_screenshots_or_weights_published"):
            self.assertIs(doc[key], False)
        content = json.dumps(doc)
        for phrase in ('"raw_prediction"', '"point"', '"screenshot_png"', '"weights"'):
            self.assertNotIn(phrase, content)

    def test_no_fabricated_real_model_generations(self):
        rows = fake_rows()
        rows[0]["real_generations"] = 0
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})
        rows = fake_rows()
        rows[1]["grammar_generate_calls"] = 0
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_invalid_output_can_never_claim_visual_or_oracle_attestation(self):
        rows = fake_rows()
        rows[1].update(
            strict_normalized_json=False,
            screenshot_integrity_attested=False,
            native_dispatched=False,
            firstparty_task_success=False,
            native_kind=None,
            all_source_visits_checked=None)
        safe = validate_rows(rows, {"free_t4": True, "devices": 1})
        self.assertEqual(safe["firstparty_task_successes"], 1)
        rows[1]["screenshot_integrity_attested"] = True
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})
        rows[1]["screenshot_integrity_attested"] = False
        rows[1]["all_source_visits_checked"] = 6
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_report_rejects_raw_private_fields_and_fake_success(self):
        for field, value in (("raw_model_text", "secret"),
                             ("absolute_point", [400, 500]),
                             ("screenshot", "sensitive")):
            with self.subTest(field=field):
                rows = fake_rows()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate_rows(rows, {"free_t4": True, "devices": 1})
        rows = fake_rows()
        rows[1]["native_dispatched"] = False
        rows[1]["firstparty_task_success"] = True
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})
        rows = fake_rows()
        rows[0]["native_kind"] = "abstain"
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_strict_boolean_and_scene_order(self):
        rows = fake_rows()
        rows[0]["native_dispatched"] = 1
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})
        rows = fake_rows()
        rows[1]["scene"] = "en_urgent_c214"
        with self.assertRaises(ValueError):
            validate_rows(rows, {"free_t4": True, "devices": 1})

    def test_fake_gpu_rejected(self):
        for bad in ({"free_t4": False, "devices": 2},
                    {"free_t4": True, "devices": True},
                    {"free_t4": True, "devices": 0},
                    {"free_t4": True, "devices": 1, "public": True}):
            with self.subTest(bad=str(bad)):
                with self.assertRaises(ValueError):
                    validate_rows(fake_rows(), bad)

    def test_real_private_gpu_environment_gate_fails_on_local_machine(self):
        class NoCuda:
            @staticmethod
            def is_available():
                return False
        class FakeTorch:
            cuda = NoCuda()
        with self.assertRaisesRegex(ValueError, "owner-private"):
            verify_free(FakeTorch)

    def test_private_kaggle_title_resolves_to_existing_verified_slug(self):
        """Avoid creating an unintended second T4 kernel on subsequent push."""
        import re
        from h282.build_private_payload import KERNEL_ID, KERNEL_TITLE

        derived = re.sub(r"[^a-z0-9]+", "-", KERNEL_TITLE.lower()).strip("-")
        self.assertEqual(KERNEL_ID, "abdulazizshehri/" + derived)
        self.assertEqual(
            KERNEL_ID,
            "abdulazizshehri/hala-h282-private-pretrained-clinicgrid-t4",
        )

    def test_code_only_kaggle_payload_no_attached_models_or_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "h282"
            src.mkdir()
            source = src / "private_clinicgrid_pretrained_probe.py"
            source.write_text(
                "from __future__ import annotations\n"
                "def validate_rows(): pass\n"
                "def run_original_clinicgrid_probe(): pass\n"
                "def stub():\n"
                "    RelativePrevalidationProcessor(1)\n"
                "    screenshot_integrity_attested = False\n"
                'if __name__ == "__main__": pass\n', encoding="utf-8")
            dest = root / "payload"
            info = build(source, dest)
            names = sorted(p.name for p in dest.iterdir())
            self.assertEqual(names,
                ["h282_private_clinicgrid_native.py", "kernel-metadata.json"])
            metadata = json.loads((dest / "kernel-metadata.json").read_text("utf-8"))
            self.assertTrue(info["private"])
            self.assertEqual(metadata["id"], "abdulazizshehri/hala-h282-private-pretrained-clinicgrid-t4")
            self.assertEqual(info["uploaded_private_media_or_weights"], 0)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            for key in ("dataset_sources", "model_sources",
                        "kernel_sources", "competition_sources"):
                self.assertEqual(metadata[key], [])
            with self.assertRaises(ValueError):
                build(source, dest)


if __name__ == "__main__":
    unittest.main()

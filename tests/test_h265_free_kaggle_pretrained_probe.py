"""H265 source-only safety contracts. No GPU, Kaggle push, media or secrets."""
import hashlib
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from h265.build_private_kaggle_payload import build_payload, SCRIPT_NAME, META_NAME, KERNEL_ID
from types import SimpleNamespace
from unittest.mock import Mock
import json
import unittest

from h265.free_kaggle_pretrained_probe import (
    APP_SHA256, SOURCE_SHA, RESULT_STATUS,
    require_free_gpu_environment, privacy_safe_measurement,
    validate_source_provenance, CountedPreloadedModel,
)

ROOT = PurePosixPath("/kaggle/working")


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True
        @staticmethod
        def device_count():
            return 1
        @staticmethod
        def get_device_name(i):
            if i != 0:
                raise AssertionError("unexpected device")
            return "Tesla T4"


class TestH265PrivateFreeGPUBaseline(unittest.TestCase):
    def test_real_gpu_guard_requires_kaggle_owner_opt_in(self):
        good = {"HALA_H265_NONCOMMERCIAL_FREE_GPU": "1",
                "KAGGLE_KERNEL_RUN_TYPE": "Batch"}
        self.assertEqual(
            require_free_gpu_environment(ROOT, environ=good, torch=FakeTorch())[
                "gpu_family"], "NVIDIA_T4")
        for env in ({}, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"},
                    {"HALA_H265_NONCOMMERCIAL_FREE_GPU": "1"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                require_free_gpu_environment(ROOT, environ=env, torch=FakeTorch())
        with self.assertRaises(ValueError):
            require_free_gpu_environment(PurePosixPath("kaggle/working"),
                                         environ=good, torch=FakeTorch())
        with self.assertRaises(ValueError):
            require_free_gpu_environment(ROOT, environ=good, torch=None)
        class PaidGPU(FakeTorch):
            class cuda(FakeTorch.cuda):
                @staticmethod
                def get_device_name(i):
                    return "NVIDIA A100"
        with self.assertRaises(ValueError):
            require_free_gpu_environment(ROOT, environ=good, torch=PaidGPU())

    def test_sha_and_git_source_fail_closed_without_upload(self):
        repo = Path(__file__).resolve().parents[1]
        app = repo / "h262" / "stockroom.html"
        self.assertEqual(hashlib.sha256(app.read_bytes()).hexdigest(), APP_SHA256)
        self.assertEqual(len(SOURCE_SHA), 40)
        good = Mock(side_effect=[SimpleNamespace(stdout=SOURCE_SHA+"\n"), SimpleNamespace(stdout="")])
        found = validate_source_provenance(repo, run=good)
        self.assertEqual(found, app)
        self.assertEqual(good.call_count, 2)
        wrong = Mock(return_value=SimpleNamespace(stdout="f"*40+"\n"))
        with self.assertRaises(ValueError):
            validate_source_provenance(repo, run=wrong)

    def test_aggregation_cannot_launder_model_or_ood_claims(self):
        env = {"free_kaggle_gpu_verified": True, "gpu_family": "NVIDIA_T4"}
        r = privacy_safe_measurement(
            generation_count=1, action_executed=True,
            action_kind="click", before=3, after=4,
            outcome="ACTION_EXECUTED", env=env)
        self.assertEqual(r["status"], RESULT_STATUS)
        self.assertTrue(r["first_party_research_task_success"])
        self.assertFalse(r["model_adapted_to_hala"])
        self.assertFalse(r["official_train_dev_ood_admitted"])
        self.assertFalse(r["independent_third_party_ood"])
        self.assertFalse(r["model_skill_qualified"])
        self.assertFalse(r["private_png_or_gold_saved"])
        self.assertNotIn("raw_model", json.dumps(r))
        with self.assertRaises(ValueError):
            privacy_safe_measurement(
                generation_count=0, action_executed=True,
                action_kind="click", before=3, after=4,
                outcome="ACTION_EXECUTED", env=env)
        with self.assertRaises(ValueError):
            privacy_safe_measurement(
                generation_count=2, action_executed=True,
                action_kind="click", before=3, after=4,
                outcome="ACTION_EXECUTED", env=env)

    def test_ambiguous_or_error_outputs_are_not_gains(self):
        env = {"free_kaggle_gpu_verified": True}
        r = privacy_safe_measurement(generation_count=1,
                action_executed=False, action_kind=None,
                before=3, after=3, outcome="INVALID_ACTION", env=env)
        self.assertFalse(r["first_party_research_task_success"])
        self.assertFalse(r["native_action_dispatched"])
        self.assertEqual(r["real_pretrained_model_generations"], 1)
        r = privacy_safe_measurement(generation_count=0,
                action_executed=False, action_kind=None,
                before=3, after=3, outcome="POLICY_OR_BROWSER_ERROR", env=env)
        self.assertFalse(r["first_party_research_task_success"])
        self.assertEqual(r["real_pretrained_model_generations"], 0)

    def test_private_payload_packaging_contains_no_assets_or_credentials(self):
        source = Path(__file__).resolve().parents[1] / "h265" / "free_kaggle_pretrained_probe.py"
        with TemporaryDirectory() as parent:
            stage = Path(parent) / "h265-private-stage"
            result = build_payload(source, stage)
            metadata = json.loads((stage / META_NAME).read_text(encoding="utf-8"))
            script = (stage / SCRIPT_NAME).read_text(encoding="utf-8")
            self.assertEqual(result["kernel_id"], KERNEL_ID)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            self.assertTrue(script.startswith("from __future__ import annotations"))
            self.assertIn("HALA_H265_NONCOMMERCIAL_FREE_GPU", script)
            self.assertIn("Qwen/Qwen3-VL-2B-Instruct", script)
            self.assertEqual({p.name for p in stage.iterdir()}, {SCRIPT_NAME, META_NAME})
            with self.assertRaises(ValueError):
                build_payload(source, stage)

    def test_completed_generate_count_not_callback_count(self):
        model = Mock()
        model.device = "cuda:0"
        model.generate.return_value = ["token"]
        counted = CountedPreloadedModel(model)
        self.assertEqual(counted.completed_generations, 0)
        counted.eval()
        counted.generate(input_ids=[1, 2])
        self.assertEqual(counted.completed_generations, 1)
        model.generate.side_effect = RuntimeError("OOM")
        with self.assertRaises(RuntimeError):
            counted.generate(input_ids=[3])
        self.assertEqual(counted.completed_generations, 1)


if __name__ == "__main__":
    unittest.main()

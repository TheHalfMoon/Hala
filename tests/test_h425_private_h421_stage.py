"""Fail-closed safety tests for H425 authentic private trained LoRA staging.

These negative fixtures ARE NOT a trained model and cannot stage/publish weights.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "h425_stage_verified_parent", ROOT / "h425" / "stage_qualified_h421.py"
)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class H425PrivateH421SourceGate(unittest.TestCase):
    def setUp(self):
        self.p = {
            "status": stage.STATUS,
            "original_source_only_gate": True,
            "stronger_source_only_gate": True,
            "trained_adapter_sha_verified": True,
            "all_eight_intermediate_checkpoint_shas_verified": True,
            "original_source_media_replayed": True,
            "actual_optimizer_steps": 12696,
            "actual_original_image_conditioned_gradient_passes": 50784,
            "receipt_sha256": "a" * 64,
        }
        self.r = {
            "status": stage.STATUS,
            "schema": "hala.h425.original-h417-replay-two-prompt-lora.v1",
            "model_weights_sha256": "b" * 64,
            "independent_external_ood": False,
            "source_rights_review_qualified": False,
            "release_qualified": False,
            "weights_public": False,
        }

    def test_every_independent_source_gate_refuses_unverified_data(self):
        for name in (
            "original_source_only_gate", "stronger_source_only_gate",
            "trained_adapter_sha_verified",
            "all_eight_intermediate_checkpoint_shas_verified",
            "original_source_media_replayed",
        ):
            with self.subTest(name=name):
                p = dict(self.p, **{name: False})
                with self.assertRaises(ValueError):
                    stage.demand_true_training(p, self.r, "a" * 64)

    def test_no_claim_of_rights_public_weights_ood_release(self):
        for name in ("independent_external_ood", "source_rights_review_qualified",
                     "release_qualified", "weights_public"):
            with self.subTest(name=name):
                r = dict(self.r, **{name: True})
                with self.assertRaises(ValueError):
                    stage.demand_true_training(self.p, r, "a" * 64)

    def test_no_forged_steps_or_provider_receipt_sha(self):
        for field, value in (
            ("actual_optimizer_steps", 12695),
            ("actual_original_image_conditioned_gradient_passes", 50780),
            ("receipt_sha256", "0" * 64),
        ):
            p = dict(self.p, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                stage.demand_true_training(p, self.r, "a" * 64)

    def test_frozen_original_qwen_lora_config_required(self):
        cfg = {
            "base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "peft_type": "LORA", "r": 8, "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        for key, mutation in (
            ("base_model_name_or_path", "other"),
            ("peft_type", "UNVERIFIED"),
            ("r", 16),
            ("lora_alpha", 32),
            ("target_modules", ["q_proj"]),
        ):
            with self.subTest(key=key):
                bad = dict(cfg, **{key: mutation})
                with self.assertRaises(ValueError):
                    stage._assert_config(bad)

    def test_missing_gpu_output_does_not_create_private_model_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "no-genuine-model-stage"
            with self.assertRaises((ValueError, FileNotFoundError)):
                stage.stage_private_h421(
                    root / "never-real-model", root / "source",
                    root / "paired", root / "h421", target
                )
            self.assertFalse(target.exists())

    def test_existing_stage_rejected_before_touching_model(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "existing"
            target.mkdir()
            marker = target / "original-owner-bytes"
            marker.write_bytes(b"must not touch")
            with self.assertRaisesRegex(ValueError, "create-only"):
                stage.stage_private_h421(
                    Path(td) / "no-provider", Path(td) / "source",
                    Path(td) / "paired", Path(td) / "h421", target
                )
            self.assertEqual(marker.read_bytes(), b"must not touch")

    def test_reject_private_model_staging_inside_git_repo(self):
        target = ROOT / "never-stage-models-inside-this-repo"
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(ValueError, "NEVER"):
            stage.stage_private_h421(
                Path("missing"), Path("missing"), Path("missing"),
                Path("missing"), target
            )
        self.assertFalse(target.exists())

    def test_no_owner_public_dataset_creation(self):
        script = (ROOT / "h425" / "stage_qualified_h421.py").read_text("utf8")
        self.assertIn('"isPrivate": True', script)
        self.assertIn('"licenses": [{"name": "other"}]', script)
        self.assertIn('H421_NAMES', script)
        self.assertIn('original_source_media_replayed', script)
        self.assertNotIn("kaggle datasets create", script)
        self.assertNotIn("subprocess.run(", script)
        self.assertNotIn("public release approval", script.lower())


if __name__ == "__main__":
    unittest.main()

"""H61 tests for the public DEV baseline receipt and the remote run wiring."""
import json
import tempfile
import unittest
from pathlib import Path

from h61.baseline import HELD_OUT_SOURCE, SCHEMA, TASK_KIND, build_baseline_receipt

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "h60" / "evidence" / "public-dev-baseline-20261021.json"


class BaselineReceiptTests(unittest.TestCase):
    def test_the_receipt_is_derived_from_the_real_h60_evidence(self):
        receipt = build_baseline_receipt(EVIDENCE)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["model_id"], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(receipt["revision"], "89644892e4d85e24eaac8bacfd4f463576704203")
        self.assertEqual(receipt["evidence"], "real_inference")
        self.assertEqual(receipt["task_kind"], TASK_KIND)
        self.assertEqual(receipt["held_out_source"], HELD_OUT_SOURCE)
        self.assertEqual(receipt["evaluated_tasks"], 48)
        self.assertEqual(receipt["verified_success_rate"], 0.0)
        self.assertFalse(receipt["independent_blind_test_completed"])
        self.assertTrue(receipt["deficit"].strip())
        self.assertEqual(len(receipt["held_out_dataset_sha256"]), 64)

    def test_the_receipt_satisfies_the_real_training_gate(self):
        from training.preflight import validate_experiment

        receipt = build_baseline_receipt(EVIDENCE)
        manifest = {
            "schema": "hala.training.manifest.v1",
            "split": "train",
            "rights_cleared": True,
            "source_kind": "rights_cleared_public",
            "dataset_sha256": "0" * 64,
            "families": ["a"],
            "held_out_families": ["b"],
        }
        config = {
            "model_id": receipt["model_id"],
            "revision": receipt["revision"],
            "precision": "fp16",
            "max_steps": 2,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 0.00005,
            "lora_rank": 8,
            "platform": "colab",
            "download_remote_weights": True,
        }
        from training.preflight import TrainingConfig

        validate_experiment(TrainingConfig.from_dict(config), manifest, receipt)

    def test_a_measurement_that_is_not_real_inference_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            evidence = json.loads(EVIDENCE.read_text("utf8"))
            evidence["model_changed"] = True
            path.write_text(json.dumps(evidence), encoding="utf8")
            with self.assertRaises(ValueError):
                build_baseline_receipt(path)


class RemoteRunWiringTests(unittest.TestCase):
    def test_the_remote_entry_point_is_importable_without_a_gpu(self):
        from h61 import colab_run, remote_entry

        self.assertTrue(callable(colab_run.main))
        self.assertTrue(callable(remote_entry.main))

    def test_the_training_config_declares_the_coordinate_supervision(self):
        config = json.loads(
            (REPO / "experiments" / "runtime" / "h61-colab-train-config.json").read_text("utf8")
        )
        from training.preflight import TrainingConfig

        parsed = TrainingConfig.from_dict(config)
        self.assertEqual(parsed.supervision, "coordinate_weighted")
        self.assertEqual(parsed.coordinate_weight, 8.0)
        self.assertEqual(parsed.platform, "colab")

    def test_an_existing_configuration_without_the_optional_fields_still_validates(self):
        from training.preflight import TrainingConfig

        parsed = TrainingConfig.from_dict(
            {
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                "precision": "fp16",
                "max_steps": 2,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "learning_rate": 0.00005,
                "lora_rank": 4,
                "platform": "colab",
                "download_remote_weights": True,
            }
        )
        self.assertEqual(parsed.supervision, "completion")

    def test_an_unknown_supervision_mode_is_refused(self):
        from training.preflight import TrainingConfig

        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(
                {
                    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                    "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                    "precision": "fp16",
                    "max_steps": 2,
                    "batch_size": 1,
                    "gradient_accumulation": 1,
                    "learning_rate": 0.00005,
                    "lora_rank": 4,
                    "platform": "colab",
                    "download_remote_weights": True,
                    "supervision": "invented",
                }
            )


if __name__ == "__main__":
    unittest.main()

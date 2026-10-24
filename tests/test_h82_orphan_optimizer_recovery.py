"""H82 CPU-only simulated atomic pointer interruption, anti-fork recovery.

Every admitted version contains full SHA-256-validated synthetic adapter and
optimizer files; no model weights, real optimizer update, GPU, or provider.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from h82.orphan_recovery import recover_orphaned_checkpoint
from training import checkpoints
from training.checkpoints import publish_checkpoint, load_checkpoint
from tests.test_training_integrity import TrainingIntegrityTests


def valid_state(step=1, total=2, dataset="b"*64):
    return {
        "config_sha256": "a"*64, "dataset_sha256": dataset,
        "manifest_sha256": "c"*64,
        "step": step, "max_steps": total,
        "metrics": [{"step": i, "loss": 1/(i+1), "seconds": 0.1} for i in range(1, step+1)],
        "changed_parameter_tensors": 2, "peak_vram_bytes": 123,
        "runtime": {"qualification_steps": [
            {"step": i, "gradient_norm_before_clip": 0.5,
             "changed_parameter_tensors_this_step": 2} for i in range(1, step+1)
        ]},
    }


def write_version(stage):
    (stage / "adapter").mkdir()
    (stage / "adapter" / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
    (stage / "resume.pt").write_bytes(b"synthetic-optimizer-state")


def interrupted_publish(root, state):
    original = checkpoints.atomic_json
    def interrupt(path, value):
        if Path(path).name == "latest.json":
            raise RuntimeError("synthetic pointer crash")
        return original(path, value)
    with mock.patch.object(checkpoints, "atomic_json", side_effect=interrupt):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "synthetic pointer"):
            publish_checkpoint(root, state, write_version)


class OrphanRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "training"
        self.expected = valid_state(1)
        self.root.mkdir()

    def test_first_optimizer_step_crash_is_recovered_without_new_training(self):
        interrupted_publish(self.root, valid_state(1))
        self.assertFalse((self.root / "latest.json").exists())
        result = recover_orphaned_checkpoint(self.root, self.expected)
        self.assertEqual(result["status"], "RECOVERED_ORPHANED_OPTIMIZER_CHECKPOINT")
        self.assertEqual(result["step"], 1)
        self.assertFalse(result["new_optimizer_step_executed"])
        self.assertFalse(result["private_provider_readback_verified"])
        self.assertEqual(load_checkpoint(self.root, self.expected)[1]["step"], 1)
        again = recover_orphaned_checkpoint(self.root, self.expected)
        self.assertEqual(again["status"], "ALREADY_PUBLISHED")

    def test_crash_after_step_two_recovers_without_fork(self):
        first = publish_checkpoint(self.root, valid_state(1), write_version)
        interrupted_publish(self.root, valid_state(2))
        self.assertEqual(load_checkpoint(self.root, self.expected)[0], first)
        result = recover_orphaned_checkpoint(self.root, self.expected)
        self.assertEqual(result["step"], 2)
        version, state = load_checkpoint(self.root, self.expected)
        self.assertEqual(state["step"], 2)
        self.assertNotEqual(version, first)
        self.assertEqual(len([p for p in (self.root / "checkpoints").iterdir()
                              if p.is_dir() and p.name.startswith("step-00000002")]), 1)

    def test_tampered_version_not_adopted_or_overwrite_previous_pointer(self):
        first = publish_checkpoint(self.root, valid_state(1), write_version)
        interrupted_publish(self.root, valid_state(2))
        orphan = next(p for p in (self.root / "checkpoints").iterdir() if "step-00000002" in p.name)
        (orphan / "resume.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "digest"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertEqual(load_checkpoint(self.root, self.expected)[0], first)

    def test_unqualified_step_without_changed_parameters_is_not_adopted(self):
        state = valid_state(1)
        state["changed_parameter_tensors"] = 0
        interrupted_publish(self.root, state)
        with self.assertRaisesRegex(ValueError, "update evidence incomplete"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertFalse((self.root / "latest.json").exists())

    def test_unqualified_step_without_gradient_history_is_not_adopted(self):
        state = valid_state(1)
        state["runtime"]["qualification_steps"] = []
        interrupted_publish(self.root, state)
        with self.assertRaisesRegex(ValueError, "update evidence incomplete"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertFalse((self.root / "latest.json").exists())

    def test_zero_gradient_step_is_not_adopted(self):
        state = valid_state(1)
        state["runtime"]["qualification_steps"][0]["gradient_norm_before_clip"] = 0
        interrupted_publish(self.root, state)
        with self.assertRaisesRegex(ValueError, "gradient evidence invalid"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertFalse((self.root / "latest.json").exists())

    def test_wrong_dataset_identity_blocks_pointer_adoption(self):
        interrupted_publish(self.root, valid_state(1))
        changed = {**self.expected, "dataset_sha256": "d"*64}
        with self.assertRaisesRegex(ValueError, "identity"):
            recover_orphaned_checkpoint(self.root, changed)
        self.assertFalse((self.root / "latest.json").exists())

    def test_ambiguous_same_step_orphans_fail_without_publish(self):
        interrupted_publish(self.root, valid_state(1))
        import shutil
        original = next((self.root / "checkpoints").iterdir())
        cloned = original.with_name("step-00000001-" + "f"*32)
        shutil.copytree(original, cloned)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertFalse((self.root / "latest.json").exists())

    def test_missing_step_is_not_promoted(self):
        interrupted_publish(self.root, valid_state(2))
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            recover_orphaned_checkpoint(self.root, self.expected)
        self.assertFalse((self.root / "latest.json").exists())

    def test_incomplete_staging_without_complete_version_does_not_resume(self):
        stages = self.root / "checkpoints"
        stages.mkdir()
        (stages / (".staging-" + "d"*32)).mkdir()
        with self.assertRaisesRegex(ValueError, "no complete"):
            recover_orphaned_checkpoint(self.root, self.expected)

    def test_symlinked_checkpoint_directory_refused(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        try:
            (self.root / "checkpoints").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Windows host disallows symlinks")
        with self.assertRaisesRegex(ValueError, "linked"):
            recover_orphaned_checkpoint(self.root, self.expected)

    def test_foreign_checkpoint_subdirectory_refused(self):
        publish_checkpoint(self.root, valid_state(1), write_version)
        (self.root / "checkpoints" / "unknown").mkdir()
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            recover_orphaned_checkpoint(self.root, self.expected)

    def test_existing_current_step_fork_rejected(self):
        publish_checkpoint(self.root, valid_state(1), write_version)
        import shutil
        original = next((self.root / "checkpoints").iterdir())
        shutil.copytree(original, original.with_name("step-00000001-" + "e"*32))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            recover_orphaned_checkpoint(self.root, self.expected)

    def test_already_published_checkpoint_is_noop(self):
        publish_checkpoint(self.root, valid_state(1), write_version)
        before = (self.root / "latest.json").read_bytes()
        result = recover_orphaned_checkpoint(self.root, self.expected)
        self.assertEqual(result["status"], "ALREADY_PUBLISHED")
        self.assertEqual((self.root / "latest.json").read_bytes(), before)

    def test_h61_completed_orphan_finalizes_without_gpu_import_or_second_update(self):
        from h61.colab_run import step_train
        from training.artifacts import sha256_file
        import os
        original = TrainingIntegrityTests()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            examples, _, _, manifest = original.artifact(work)
            config, paths = original.write_runner_inputs(work, manifest)
            config["max_steps"] = 1
            paths[0].write_text(json.dumps(config), "utf8")
            # H61's remote entry expects TRAIN artifacts in work/export,
            # not the fixture's generic authoring root. Preserve bytes
            # exactly so the original manifest and dataset SHA still match.
            import shutil
            export = work / "export"
            (export / "images").mkdir(parents=True)
            shutil.copy2(examples, export / "h61_drag_curriculum.jsonl")
            shutil.copy2(paths[1], export / "h61_drag_curriculum_manifest.json")
            shutil.copy2(paths[2], work / "public-dev-baseline-receipt.json")
            shutil.copy2(work / "images" / "a.png", export / "images" / "a.png")
            output = work / "training"
            output.mkdir()
            state = valid_state(1, total=1)
            state.update(
                config_sha256=hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()).hexdigest(),
                dataset_sha256=manifest["dataset_sha256"],
                manifest_sha256=sha256_file(paths[1]),
                max_steps=1,
            )
            interrupted_publish(output, state)
            self.assertFalse((output / "latest.json").exists())
            # H61 must identify the checkpoint-version directory and route
            # back into resume, not refuse the output as a fresh run.
            with (mock.patch("training.scripts.train_adapter.platform.system",
                             return_value="Linux"),
                  mock.patch.dict(os.environ, {"HALA_REMOTE_TRAINING_ALLOWED": "1"}),
                  mock.patch("training.scripts.train_adapter.probe") as gpu_probe):
                result = step_train(work, paths[0], 1, False)
                gpu_probe.assert_not_called()
            self.assertEqual(result["changed_parameter_tensors"], 2)
            self.assertEqual(load_checkpoint(output, state)[1]["step"], 1)
            self.assertEqual(len([p for p in (output / "checkpoints").iterdir()
                                 if p.name.startswith("step-00000001")]), 1)

    def test_h61_enters_resume_if_only_orphan_versions_exist(self):
        from h61.colab_run import step_train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "export").mkdir()
            conf = root / "config.json"
            conf.write_text(json.dumps({"model_id": "Qwen/Qwen3-VL-2B-Instruct"}), "utf8")
            (root / "training" / "checkpoints").mkdir(parents=True)
            with mock.patch("training.scripts.train_adapter.train",
                            return_value={"step": 1}) as trainer:
                step_train(root, conf, 1, False)
            self.assertTrue(trainer.call_args.kwargs["resume"])


if __name__ == "__main__":
    unittest.main()

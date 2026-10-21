"""H60 matched adapter evaluation: checkpoint admission tests.

The adapter evaluation is only allowed to load a checkpoint that the training
run published through the atomic pointer and whose every artifact digest
still matches. These tests build a synthetic checkpoint bundle in the exact
published layout and prove that:

* a complete, untampered bundle is admitted and its identity is reported;
* a tampered adapter tensor is rejected;
* a pointer that does not match the checkpoint metadata digest is rejected;
* a checkpoint that records no changed adapter parameter is rejected.

No model, no GPU and no provider account is involved.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h60.real_drag_dev_adapter_eval import verify_adapter_checkpoint
from training.artifacts import sha256_file
from training.checkpoints import IDENTITY_FIELDS, file_hashes


def _state(step: int = 3, max_steps: int = 3, changed: int = 2) -> dict:
    return {
        **{name: "a" * 64 for name in IDENTITY_FIELDS},
        "step": step,
        "max_steps": max_steps,
        "metrics": [
            {"step": index, "loss": 1.0 / index, "seconds": 0.5}
            for index in range(1, step + 1)
        ],
        "changed_parameter_tensors": changed,
        "peak_vram_bytes": 12345678,
        "runtime": {"gpu": "Tesla T4"},
    }


def _write_bundle(root: Path, *, changed: int = 2) -> Path:
    version = root / "checkpoints" / ("step-%08d-%s" % (3, "b" * 32))
    (version / "adapter").mkdir(parents=True)
    (version / "adapter" / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
    (version / "resume.pt").write_bytes(b"fake-resume-state")
    artifacts = file_hashes(version)
    metadata = {
        "schema": "hala.training.checkpoint.v1",
        "state": _state(changed=changed),
        "artifacts": artifacts,
    }
    (version / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    (root / "latest.json").write_text(
        json.dumps(
            {
                "schema": "hala.training.latest.v1",
                "version": version.relative_to(root).as_posix(),
                "checkpoint_sha256": sha256_file(version / "checkpoint.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    return version


class H60AdapterCheckpointAdmissionTests(unittest.TestCase):
    def test_a_complete_checkpoint_is_admitted_with_its_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lora"
            version = _write_bundle(root)
            identity = verify_adapter_checkpoint(str(root))
            self.assertEqual(identity["checkpoint_version"], version.name)
            self.assertEqual(len(identity["checkpoint_sha256"]), 64)
            self.assertEqual(identity["checkpoint_step"], 3)
            self.assertEqual(identity["changed_parameter_tensors"], 2)
            self.assertIn("adapter/adapter_model.safetensors", identity["adapter_files"])
            self.assertEqual(
                identity["adapter_file_sha256"]["adapter/adapter_model.safetensors"],
                sha256_file(version / "adapter" / "adapter_model.safetensors"),
            )
            self.assertEqual(identity["training_identity"]["dataset_sha256"], "a" * 64)
            self.assertAlmostEqual(identity["last_step_loss"], 1.0 / 3)

    def test_a_tampered_adapter_tensor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lora"
            version = _write_bundle(root)
            (version / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered-weights")
            with self.assertRaises(ValueError):
                verify_adapter_checkpoint(str(root))

    def test_a_pointer_that_does_not_match_the_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lora"
            _write_bundle(root)
            pointer = json.loads((root / "latest.json").read_text("utf8"))
            pointer["checkpoint_sha256"] = "b" * 64
            (root / "latest.json").write_text(json.dumps(pointer), encoding="utf8")
            with self.assertRaises(ValueError):
                verify_adapter_checkpoint(str(root))

    def test_a_checkpoint_without_a_changed_parameter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lora"
            _write_bundle(root, changed=0)
            with self.assertRaises(ValueError):
                verify_adapter_checkpoint(str(root))

    def test_a_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                verify_adapter_checkpoint(str(Path(temporary) / "absent"))


if __name__ == "__main__":
    unittest.main()

"""H60 matched adapter evaluation over the first-party PUBLIC DEV drag scenes.

This runner is the *matched* counterpart of the frozen-base run
`h60.real_drag_dev_baseline`:

* it **calls that runner itself**, so the case loader, the prompt text, the
  prompt field allowlist, the screenshot provenance checks, the
  published-answer leak guard, the generation settings and the raw-attempts
  schema are identical by construction rather than by copy;
* the only intended difference is the model: the pinned unchanged base plus
  one verified LoRA adapter checkpoint;
* the adapter checkpoint is admitted through the training run's own atomic
  pointer before any weight is loaded. The pointer, the checkpoint metadata
  digest, every artifact digest and the adapter tensor file are checked, and
  their digests are recorded in the receipt.

This module never reads a DEV answer file, never uses a published answer
coordinate to build a prompt, never trains, and never touches official
benchmark TEST or any external account. It runs only on an explicitly
approved free Google Colab Linux CUDA session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h60.real_drag_dev_baseline import canonical, run as run_public_dev
from training.checkpoints import IDENTITY_FIELDS, load_checkpoint, validate_version

ADAPTER_SCHEMA = "hala.h60.real-adapter-qwen-native-drag-public-dev.v1"


def verify_adapter_checkpoint(output: str) -> dict:
    """Verify the published checkpoint bundle and return its identity.

    The bundle is admitted only through the atomic pointer the training run
    published. Every artifact digest is recomputed from the bytes on disk, so
    a tampered adapter tensor can never be loaded.
    """
    root = Path(output)
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        raise ValueError("A published training checkpoint is required")
    pointer = json.loads(pointer_path.read_text("utf8"))
    version, state = load_checkpoint(root, None)
    metadata_digest = canonical_digest(version / "checkpoint.json")
    if pointer.get("checkpoint_sha256") != metadata_digest:
        raise ValueError("Published checkpoint pointer does not match the checkpoint digest")
    state, artifacts = validate_version(version, None)
    adapter_files = sorted(name for name in artifacts if name.startswith("adapter/"))
    if not adapter_files:
        raise ValueError("The checkpoint carries no adapter weights")
    weights = [name for name in adapter_files if name.endswith("adapter_model.safetensors")]
    if not weights:
        raise ValueError("The checkpoint carries no adapter weight tensor")
    if not state.get("changed_parameter_tensors"):
        raise ValueError("The checkpoint records no changed adapter parameter tensor")
    metrics = state.get("metrics") or []
    if not metrics:
        raise ValueError("The checkpoint records no training metric")
    return {
        "checkpoint_version": version.name,
        "checkpoint_sha256": metadata_digest,
        "checkpoint_step": state["step"],
        "checkpoint_max_steps": state["max_steps"],
        "changed_parameter_tensors": state["changed_parameter_tensors"],
        "peak_vram_bytes": state["peak_vram_bytes"],
        "adapter_dir": str(version / "adapter"),
        "adapter_files": adapter_files,
        "adapter_file_sha256": {name: artifacts[name] for name in adapter_files},
        "adapter_weights_sha256": artifacts[weights[0]],
        "training_identity": {name: state[name] for name in IDENTITY_FIELDS},
        "first_step_loss": metrics[0]["loss"],
        "last_step_loss": metrics[-1]["loss"],
        "training_seconds_total": round(sum(row["seconds"] for row in metrics), 3),
        "runtime": state.get("runtime"),
    }


def canonical_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    root: str,
    output: str,
    checkpoint: str,
    *,
    dev: str = "h55",
    max_cases: int = 24,
):
    """Run the identical public-dev protocol with a verified adapter."""
    identity = verify_adapter_checkpoint(checkpoint)
    result = run_public_dev(
        root,
        output,
        dev=dev,
        max_cases=max_cases,
        adapter=identity["adapter_dir"],
        adapter_sha256=identity["adapter_weights_sha256"],
    )
    receipt_path = Path(output) / "run-result.json"
    receipt = json.loads(receipt_path.read_text("utf8"))
    receipt["schema"] = ADAPTER_SCHEMA
    receipt["model_training_executed"] = True
    receipt["adapters_loaded"] = True
    receipt["adapter_checkpoint"] = identity
    receipt["matched_to_frozen_base_runner"] = "h60.real_drag_dev_baseline.run"
    receipt_path.write_bytes(canonical(receipt))
    partial = Path(output) / "run-partial.json"
    if partial.exists():
        partial.write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dev", required=True, choices=["h55", "h58"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=24)
    args = parser.parse_args()
    result = run(
        args.source,
        args.output,
        args.checkpoint,
        dev=args.dev,
        max_cases=args.max_cases,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "source_cases",
                    "max_cases",
                    "model_id",
                    "revision",
                    "dev_set",
                    "adapters_loaded",
                    "raw_attempts_sha256",
                )
            },
            sort_keys=True,
        )
    )

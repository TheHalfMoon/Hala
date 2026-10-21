"""H60 real resume-equivalence qualification on authentic private DragOn TRAIN.

The H15 qualification proved the save/reload/resume machinery on a synthetic
58-row candidate. H60 proves the same machinery on the **authentic private
original DragOn TRAIN** source: one continuous trajectory and one deliberately
interrupted-and-resumed trajectory must end at bit-identical adapter weights,
optimizer state, gradient scaler state, CPU and CUDA RNG state and loss
trajectory.

This is a runtime recoverability proof on real data. It is not a model
quality qualification and it never claims one.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

from training.artifacts import sha256_file
from training.checkpoints import atomic_json, load_checkpoint, validate_version
from training.scripts.qualify_lora import (
    checkpoint_evidence,
    compare_values,
    release_runtime,
    snapshot_state,
)
from training.scripts.train_adapter import QualificationInterrupted, train

SCHEMA = "hala.h60.real-dragon-lora-resume-qualification.v1"


def compare_trajectories(continuous: Path, resumed: Path, expected_step: int) -> dict:
    """Compare two published trajectories exactly, as the H15 harness does."""
    import torch
    from safetensors.torch import load_file

    first, a = snapshot_state(continuous)
    second, b = snapshot_state(resumed)
    for key in ("config_sha256", "dataset_sha256", "manifest_sha256", "max_steps"):
        if a[key] != b[key]:
            raise ValueError("Resume trajectory identity mismatch")
    if a["step"] != expected_step or b["step"] != expected_step:
        raise ValueError("Resume trajectory did not reach the expected final step")
    adapter_a = load_file(str(first / "adapter/adapter_model.safetensors"))
    adapter_b = load_file(str(second / "adapter/adapter_model.safetensors"))
    state_a = torch.load(first / "resume.pt", map_location="cpu", weights_only=True)
    state_b = torch.load(second / "resume.pt", map_location="cpu", weights_only=True)
    if state_a.keys() != state_b.keys():
        raise ValueError("Resume snapshot field mismatch")
    comparisons = {"adapter": compare_values(adapter_a, adapter_b)}
    comparisons["loss_trajectory"] = compare_values(
        [{"step": row["step"], "loss": row["loss"]} for row in a["metrics"]],
        [{"step": row["step"], "loss": row["loss"]} for row in b["metrics"]],
    )
    for name in state_a:
        strict = name in ("torch_rng", "cuda_rng", "scaler")
        comparisons[name] = compare_values(
            state_a[name], state_b[name], atol=0 if strict else 1e-6, rtol=0 if strict else 1e-5
        )
    return {
        "exact": all(row["exact"] for row in comparisons.values()),
        "within_tolerance": all(row["within_tolerance"] for row in comparisons.values()),
        "components": comparisons,
    }


def qualify(
    config: str,
    examples: str,
    manifest: str,
    baseline: str,
    output: str,
    *,
    total_steps: int = 6,
    interrupt_step: int = 3,
) -> dict:
    if platform.system() != "Linux" or os.environ.get("HALA_REMOTE_TRAINING_ALLOWED") != "1":
        raise ValueError("Explicit remote Linux opt-in required; local weights forbidden")
    if not 1 <= interrupt_step < total_steps:
        raise ValueError("The interruption must leave a remaining optimizer step")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh qualification output required")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"config": config, "examples": examples, "manifest": manifest, "baseline": baseline}
    input_hashes = {name: sha256_file(Path(path)) for name, path in inputs.items()}

    def guard_inputs() -> None:
        if any(sha256_file(Path(path)) != input_hashes[name] for name, path in inputs.items()):
            raise ValueError("Qualification input identity changed between trajectories")

    started = time.perf_counter()
    continuous = output / "continuous"
    resumed = output / "resumed"
    train(config, examples, manifest, baseline, continuous)
    release_runtime()
    guard_inputs()
    interrupted = False
    try:
        train(config, examples, manifest, baseline, resumed, stop_after_step=interrupt_step)
    except QualificationInterrupted as stopped:
        if stopped.step != interrupt_step:
            raise ValueError("Wrong interruption step") from None
        interrupted = True
    release_runtime()
    if not interrupted:
        raise ValueError("Required deliberate interruption did not occur")
    _, partial = snapshot_state(resumed)
    if partial["step"] != interrupt_step or partial["max_steps"] != total_steps:
        raise ValueError("Required partial snapshot is missing")
    first_snapshot = checkpoint_evidence(resumed)
    guard_inputs()
    train(config, examples, manifest, baseline, resumed, resume=True)
    release_runtime()
    guard_inputs()
    comparison = compare_trajectories(continuous, resumed, total_steps)
    continuous_evidence = checkpoint_evidence(continuous)
    resumed_evidence = checkpoint_evidence(resumed)
    reloaded = (
        resumed_evidence.get("runtime", {}).get("actual_reload") is True
        and resumed_evidence.get("runtime", {}).get("resumed_from_step") == interrupt_step
    )
    report = {
        "schema": SCHEMA,
        "evidence": "real_gpu_engineering_qualification",
        "source": "authentic private original Hcompany DragOn TRAIN (H57 verified shard)",
        "runtime_qualified": bool(reloaded and comparison["within_tolerance"]),
        "adapter_quality_qualified": False,
        "independent_blind_test_completed": False,
        "official_benchmark_comparison_qualified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": input_hashes,
        "total_steps": total_steps,
        "interrupt_step": interrupt_step,
        "first_partial_snapshot": first_snapshot,
        "continuous": continuous_evidence,
        "resumed": resumed_evidence,
        "comparison": comparison,
        "consumed_microbatches_per_trajectory": total_steps,
        "trajectories": 2,
    }
    atomic_json(output / "qualification-result.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "examples", "manifest", "baseline", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--total-steps", type=int, default=6)
    parser.add_argument("--interrupt-step", type=int, default=3)
    args = parser.parse_args()
    result = qualify(
        args.config,
        args.examples,
        args.manifest,
        args.baseline,
        args.output,
        total_steps=args.total_steps,
        interrupt_step=args.interrupt_step,
    )
    print(json.dumps({
        "schema": result["schema"],
        "runtime_qualified": result["runtime_qualified"],
        "exact": result["comparison"]["exact"],
        "within_tolerance": result["comparison"]["within_tolerance"],
        "elapsed_seconds": round(result["elapsed_seconds"], 3),
    }, sort_keys=True))

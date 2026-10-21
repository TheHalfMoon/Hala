"""H60 matched adapter evaluation over the first-party PUBLIC DEV drag scenes.

This runner is the *matched* counterpart of `h60.real_drag_dev_baseline`:

* it reuses that module's case loader, so the prompt text, the prompt field
  allowlist, the screenshot provenance checks and the published-answer leak
  guard are byte-identical to the frozen-base run;
* it reuses that module's generation settings and the same create-only output
  rules;
* it writes the same raw-attempts schema, so the **same** H55/H58 scorer
  measures the adapter and the frozen base.

The only intended difference is the model: the pinned unchanged base plus one
verified LoRA adapter checkpoint. The adapter checkpoint is verified before
the weights are loaded: the atomic pointer, the checkpoint metadata digest,
every artifact digest and the adapter tensor file are all checked, and their
digests are recorded in the receipt.

This module never reads a DEV answer file, never uses a published answer
coordinate to build a prompt, never trains, and never touches official
benchmark TEST or any external account. It runs only on an explicitly
approved free Google Colab Linux CUDA session.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from h60.real_drag_dev_baseline import (
    ATTEMPTS_SCHEMA,
    DEV_SETS,
    MAX_CASES,
    MODEL_SCHEMA,
    PROMPT_REVISION,
    PROTOCOL,
    _source_inputs,
    canonical,
    sha,
    validate_remote,
)
from training.checkpoints import IDENTITY_FIELDS, load_checkpoint, validate_version
from training.scripts.baseline import MODEL_ID, MODEL_REVISION, reject_kaggle_runtime

ADAPTER_SCHEMA = "hala.h60.real-adapter-qwen-native-drag-public-dev.v1"


def verify_adapter_checkpoint(output: str) -> dict:
    """Verify the published checkpoint bundle and return its identity."""
    root = Path(output)
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        raise ValueError("A published training checkpoint is required")
    pointer = json.loads(pointer_path.read_text("utf8"))
    version, state = load_checkpoint(root, None)
    metadata_digest = sha((version / "checkpoint.json").read_bytes())
    if pointer.get("checkpoint_sha256") != metadata_digest:
        raise ValueError("Published checkpoint pointer does not match the checkpoint digest")
    state, artifacts = validate_version(version, None)
    adapter_files = sorted(name for name in artifacts if name.startswith("adapter/"))
    if not adapter_files:
        raise ValueError("The checkpoint carries no adapter weights")
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
        "adapter_files": adapter_files,
        "adapter_file_sha256": {name: artifacts[name] for name in adapter_files},
        "training_identity": {name: state[name] for name in IDENTITY_FIELDS},
        "first_step_loss": metrics[0]["loss"],
        "last_step_loss": metrics[-1]["loss"],
        "training_seconds_total": round(sum(row["seconds"] for row in metrics), 3),
        "runtime": state.get("runtime"),
    }


def run(
    root: str,
    output: str,
    checkpoint: str,
    *,
    dev: str = "h55",
    max_cases: int = 24,
    max_new_tokens: int = 160,
):
    validate_remote()
    reject_kaggle_runtime()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle GPU training/inference not authorized")
    if dev not in DEV_SETS:
        raise ValueError("Unknown public DEV set")
    if type(max_cases) is not int or not 1 <= max_cases <= MAX_CASES:
        raise ValueError("Bounded actual model run parameters required")
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 192:
        raise ValueError("Bounded actual model run parameters required")

    identity = verify_adapter_checkpoint(checkpoint)
    cases = _source_inputs(Path(root), dev)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Create-only adapter evaluation output folder")
    output.mkdir(parents=True)

    import torch
    import transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or transformers.__version__ != "5.16.1":
        raise ValueError("Unqualified installed CUDA/transformers. No silent fallback")
    gpu = torch.cuda.get_device_properties(0)
    info = {
        "schema": ADAPTER_SCHEMA,
        "dev_set": dev,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "evidence": "real_inference",
        "source_sha256": sha((Path(root) / "source.json").read_bytes()),
        "coordinate_protocol": PROTOCOL,
        "split": "public_dev",
        "blind_test": False,
        "model_training_executed": True,
        "adapters_loaded": True,
        "provider": "interactive_approved_free_colab",
        "gpu": gpu.name,
        "vram_bytes": gpu.total_memory,
        "package_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "adapter_checkpoint": identity,
        "real_generated_cases": [],
        "source_cases": len(cases),
        "max_cases": max_cases,
        "status": "loading_model",
        "inference_failure": None,
        "independent_model_performance_qualified": False,
    }

    def save_partial() -> None:
        (output / "run-partial.json").write_bytes(canonical(info))

    save_partial()
    model = None
    try:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["HF_HOME"] = "/tmp/hala-h60-model-cache"
        torch.manual_seed(42)
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to("cuda:0")
        adapter_dir = Path(checkpoint) / identity["checkpoint_version"] / "adapter"
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        model = model.eval()
        info["status"] = "inferencing"
        save_partial()
        for scene in cases[:max_cases]:
            with Image.open(scene["image_path"]) as image:
                if image.format != "PNG" or image.size != (1024, 640):
                    raise ValueError("Source screenshot changed since frozen preflight")
                picture = image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": scene["prompt"]}],
                }
            ]
            template = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[template], images=[picture], return_tensors="pt").to("cuda:0")
            before = time.perf_counter()
            with torch.inference_mode():
                token_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, max_time=60
                )
            torch.cuda.synchronize(0)
            generated = token_ids[:, inputs.input_ids.shape[1] :]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            if not isinstance(raw, str) or len(raw) > 8192:
                raise ValueError("Unbounded original neural text output")
            info["real_generated_cases"].append(
                {
                    "id": scene["id"],
                    "image_sha256": scene["image_sha256"],
                    "input_prompt_sha256": sha(scene["prompt"].encode("utf8")),
                    "language": scene["language"],
                    "kind": scene["kind"],
                    "ordered": scene["ordered"],
                    "output_raw": raw,
                    "output_sha256": sha(raw.encode("utf8")),
                    "elapsed_seconds": time.perf_counter() - before,
                    "output_tokens": int(generated.shape[1]),
                    "status": "real_neural_raw",
                }
            )
            save_partial()
            print(
                "H60_ACTUAL_ADAPTER_RAW_CASE",
                len(info["real_generated_cases"]),
                scene["id"],
                flush=True,
            )
            del inputs, token_ids, generated
        info["status"] = "completed" if max_cases == MAX_CASES else "diagnostic_partial_not_baseline"
        info["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(0))
        if max_cases == MAX_CASES:
            predictions = {
                "schema": ATTEMPTS_SCHEMA[dev],
                "source_sha256": info["source_sha256"],
                "origin": "provider_raw_model_outputs",
                "cases": [
                    {
                        "id": case["id"],
                        "image_sha256": case["image_sha256"],
                        "raw_output": case["output_raw"],
                    }
                    for case in info["real_generated_cases"]
                ],
            }
            (output / "raw-attempts.json").write_bytes(canonical(predictions))
            info["raw_attempts_sha256"] = sha((output / "raw-attempts.json").read_bytes())
        (output / "run-result.json").write_bytes(canonical(info))
        return info
    except Exception as exc:  # noqa: BLE001
        info["status"] = "blocked"
        info["inference_failure"] = type(exc).__name__
        save_partial()
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dev", required=True, choices=sorted(DEV_SETS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=MAX_CASES)
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
                    "raw_attempts_sha256",
                )
            },
            sort_keys=True,
        )
    )

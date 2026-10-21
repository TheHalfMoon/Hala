"""H60 genuine unchanged-Qwen native drag PUBLIC DEV baseline for H55 and H58.

Runs the frozen, unchanged pinned Qwen3-VL-2B over the first-party public DEV
drag scenes and records the raw neural generations. This module never reads a
DEV answer file, never uses a published answer coordinate to build a prompt,
never trains, and never touches official benchmark TEST or any external
account. It runs only on an explicitly approved free Google Colab Linux CUDA
session, exactly like the reviewed H56 runner it generalises.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import time
from pathlib import Path

from training.scripts.baseline import MODEL_ID, MODEL_REVISION, reject_kaggle_runtime

DEV_SETS = {
    "h55": {
        "schema": "hala.h55.original-bilingual-drag-public-dev.v1",
        "source_sha256": "707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6",
    },
    "h58": {
        "schema": "hala.h58.original-multidomain-public-dev.v1",
        "source_sha256": "a1c17c5dca028096bdd653de927fae18b1797a241ca3529646e10a1eb5f4f3de",
    },
}
MAX_CASES = 24
PROTOCOL = "original_image_pixels_1024x640_native_drag_v1"
PROMPT_REVISION = "hala-h60-native-drag-public-dev-prompt-v1"
MODEL_SCHEMA = "hala.h60.real-unchanged-qwen-native-drag-public-dev.v1"
ATTEMPTS_SCHEMA = {
    "h55": "hala.h55.public-dev-raw-drag-attempts.v1",
    "h58": "hala.h58.raw-public-dev-native-drag-attempts.v1",
}
# Only these plan fields may ever influence a model prompt.
PROMPT_FIELDS = ("id", "language", "family", "kind", "ordered", "instruction", "title",
                 "goal_percent", "image_size", "viewport", "split")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def _source_inputs(root: Path, dev: str) -> list[dict]:
    """Read only prompt inputs and original provenance. DEV answers are never read."""
    spec = DEV_SETS[dev]
    source = json.loads((root / "source.json").read_text("utf8"))
    plans = json.loads((root / "plans.json").read_text("utf8"))
    raw_source = (root / "source.json").read_bytes()
    if (
        sha(raw_source) != spec["source_sha256"]
        or source.get("schema") != spec["schema"]
        or source.get("public_dev_only") is not True
        or source.get("blind_test") is not False
        or source.get("split") != "dev"
        or source.get("plans_sha256") != sha((root / "plans.json").read_bytes())
        or len(plans) != MAX_CASES
        or len(source.get("scenes", [])) != MAX_CASES
    ):
        raise ValueError("Frozen first-party PUBLIC DEV provenance mismatch")
    rights = source.get("rights", {})
    if rights.get("train_eligible") is True or rights.get("training_allowed") is True:
        raise ValueError("PUBLIC DEV assets must never be training eligible")
    cases: list[dict] = []
    image_digests: set[str] = set()
    languages = {"ar": 0, "en": 0}
    families: set[str] = set()
    for plan, entry in zip(plans, source["scenes"]):
        ident = plan["id"]
        if entry.get("id") != ident or plan.get("split") != "dev":
            raise ValueError("Unexpected public DEV plan or scene metadata")
        size = plan.get("viewport") or plan.get("image_size")
        if size != [1024, 640]:
            raise ValueError("Unexpected public DEV screenshot size")
        language = plan.get("language")
        if language not in languages:
            raise ValueError("Unexpected public DEV language")
        html_path = root / "html" / (ident + ".html")
        image_path = root / "png" / (ident + ".png")
        if sha(html_path.read_bytes()) != entry["html_sha256"]:
            raise ValueError("Authored public DEV source HTML changed")
        png = image_path.read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 128 or len(png) > 5_000_000:
            raise ValueError("Original real Edge screenshot is not a bounded PNG")
        if struct.unpack(">II", png[16:24]) != (1024, 640):
            raise ValueError("Incorrect original screenshot dimensions")
        image_hash = sha(png)
        if image_hash in image_digests:
            raise ValueError("Duplicate original screenshot")
        image_digests.add(image_hash)
        languages[language] += 1
        families.add(plan.get("family", ""))
        ordered = bool(plan.get("ordered", True))
        if dev == "h55":
            if type(plan.get("goal_percent")) is not int or not 15 <= plan["goal_percent"] <= 95:
                raise ValueError("Unexpected H55 public DEV slider plan")
            body = (
                "اسحب مؤشر الواجهة الافتراضية إلى " + str(plan["goal_percent"]) + "٪. "
                "أعد فعل سحب واحد فقط بإحداثيات الصورة الأصلية."
                if language == "ar"
                else "Drag the fictional interface slider to " + str(plan["goal_percent"])
                + "%. Return only ONE ordered drag action in original-image pixel coordinates."
            )
        else:
            instruction = plan.get("instruction")
            if not isinstance(instruction, str) or not 4 <= len(instruction) <= 400:
                raise ValueError("Unexpected H58 public DEV instruction")
            body = (
                instruction + "\nأعد فعل سحب واحد فقط بإحداثيات الصورة الأصلية."
                if language == "ar"
                else instruction + "\nReturn only ONE drag action in original-image pixel coordinates."
            )
        prompt = body + (
            "\nThe image is 1024x640 pixels. Return ONLY a JSON object with type=drag, "
            "start_point=[x,y], end_point=[x,y], coordinate_space=source_image_pixels, "
            "ordered=" + ("true" if ordered else "false") + ". "
            "Coordinates must be integer pixels of the ORIGINAL screenshot; "
            "do not click, navigate, or execute anything."
        )
        # Hard guard: a published answer coordinate must never reach the prompt.
        for forbidden in ("start", "end", "bad_start", "bad_end"):
            value = plan.get(forbidden)
            if isinstance(value, list) and len(value) == 2:
                for coordinate in value:
                    if isinstance(coordinate, int) and f"[{coordinate}," in prompt:
                        raise ValueError("Published answer coordinate leaked into the prompt")
        cases.append(
            {
                "id": ident,
                "language": language,
                "family": plan.get("family", ""),
                "kind": plan.get("kind", "slider"),
                "ordered": ordered,
                "image_path": str(image_path),
                "image_sha256": image_hash,
                "prompt": prompt,
            }
        )
    if languages != {"ar": 12, "en": 12} or len(families) < 3:
        raise ValueError("Missing full disjoint bilingual public DEV inventory")
    return cases


def validate_remote() -> None:
    if (
        platform.system() != "Linux"
        or not os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("HALA_REMOTE_BASELINE_ALLOWED") != "1"
    ):
        raise ValueError(
            "Only explicitly authorized interactive FREE Colab Linux session may run weights"
        )
    reject_kaggle_runtime()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle GPU training/inference not authorized")
    if not Path("/content").exists():
        raise ValueError("Missing interactive Colab workspace; no remote weights elsewhere")


def run(root: str, output: str, *, dev: str = "h55", max_cases: int = 24, max_new_tokens: int = 160):
    validate_remote()
    if dev not in DEV_SETS:
        raise ValueError("Unknown public DEV set")
    if type(max_cases) is not int or not 1 <= max_cases <= MAX_CASES:
        raise ValueError("Bounded actual model run parameters required")
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 192:
        raise ValueError("Bounded actual model run parameters required")
    cases = _source_inputs(Path(root), dev)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Create-only baseline output folder")
    output.mkdir(parents=True)
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or transformers.__version__ != "5.16.1":
        raise ValueError("Unqualified installed CUDA/transformers. No silent fallback")
    gpu = torch.cuda.get_device_properties(0)
    info = {
        "schema": MODEL_SCHEMA,
        "dev_set": dev,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "evidence": "real_inference",
        "source_sha256": sha((Path(root) / "source.json").read_bytes()),
        "coordinate_protocol": PROTOCOL,
        "split": "public_dev",
        "blind_test": False,
        "model_training_executed": False,
        "adapters_loaded": False,
        "provider": "interactive_approved_free_colab",
        "gpu": gpu.name,
        "vram_bytes": gpu.total_memory,
        "package_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
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
        ).to("cuda:0").eval()
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
                "H60_ACTUAL_QWEN_RAW_CASE",
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=MAX_CASES)
    args = parser.parse_args()
    result = run(args.source, args.output, dev=args.dev, max_cases=args.max_cases)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "source_cases", "max_cases", "model_id", "revision", "dev_set")
            },
            sort_keys=True,
        )
    )

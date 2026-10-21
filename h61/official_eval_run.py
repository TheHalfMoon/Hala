"""H61 the real model run over the upstream official public evaluation split.

The runner is the same shape as the H60 public DEV runner: it loads the pinned
unchanged model, optionally loads one verified adapter checkpoint, generates
the raw text, preserves every raw string verbatim and never reads an answer
into a prompt. The prompt is the one shared H61 contract, and the scorer is
the explicit geometry rule of `h61.official_eval`.

It runs only on an explicitly approved free Colab Linux CUDA session.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from h61.official_eval import SCHEMA, apply_frame, canonical, fetch_domain, load_scenes, score

RUN_SCHEMA = "hala.h61.official-dragon-eval-run.v1"
ATTEMPTS_SCHEMA = "hala.h61.official-dragon-eval-raw-attempts.v1"
MAX_SCENES = 250


def validate_remote() -> None:
    if (
        platform.system() != "Linux"
        or not os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("HALA_REMOTE_BASELINE_ALLOWED") != "1"
    ):
        raise ValueError(
            "Only an explicitly approved interactive FREE Colab Linux session may run weights"
        )
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle GPU training/inference is not authorized")
    if not Path("/content").exists():
        raise ValueError("Missing interactive Colab workspace; no remote weights elsewhere")


def sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def run(domains: list[str], work: str, *, limit: int = 60, max_new_tokens: int = 160,
        adapter: str | None = None, adapter_sha256: str | None = None) -> dict:
    """Run the pinned model over the official evaluation scenes of every domain."""
    validate_remote()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    from training.scripts.baseline import MODEL_ID, MODEL_REVISION, reject_kaggle_runtime

    reject_kaggle_runtime()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or transformers.__version__ != "5.16.1":
        raise ValueError("Unqualified installed CUDA/transformers. No silent fallback")
    gpu = torch.cuda.get_device_properties(0)

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_HOME"] = "/tmp/hala-h61-model-cache"
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                             trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False,
                                           trust_remote_code=False).eval()

    receipt = {
        "schema": RUN_SCHEMA,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "provider": "interactive_approved_free_colab",
        "gpu": gpu.name,
        "vram_bytes": gpu.total_memory,
        "package_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "adapters_loaded": adapter is not None,
        "adapter_sha256": adapter_sha256,
        "evidence": "real_inference",
        "domains": {},
    }
    for domain in domains:
        domain_dir = work / domain
        scenes_path = domain_dir / "scenes.json"
        if not scenes_path.exists():
            fetch_domain(domain, limit, domain_dir)
        scenes = load_scenes(scenes_path)
        scenes = apply_frame(scenes, domain_dir)
        outputs = {}
        records = []
        for index, scene in enumerate(scenes):
            with Image.open(scene["image_path"]) as image:
                picture = image.convert("RGB")
            messages = [
                {"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": scene["prompt"]}]}
            ]
            template = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[template], images=[picture], return_tensors="pt").to("cuda:0")
            before = time.perf_counter()
            with torch.inference_mode():
                token_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False, max_time=60)
            torch.cuda.synchronize(0)
            generated = token_ids[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            outputs[scene["id"]] = raw
            records.append(
                {
                    "id": scene["id"],
                    "image_sha256": scene["image_sha256"],
                    "input_prompt_sha256": sha(scene["prompt"].encode("utf8")),
                    "raw_output": raw,
                    "elapsed_seconds": time.perf_counter() - before,
                }
            )
            del inputs, token_ids, generated
            print("H61_OFFICIAL_CASE", domain, index + 1, len(scenes), flush=True)
        scored = score(scenes, outputs)
        (domain_dir / "raw-attempts.json").write_bytes(
            canonical(
                {
                    "schema": ATTEMPTS_SCHEMA,
                    "origin": "provider_raw_model_outputs",
                    "source": "Hcompany/DragOn eval split",
                    "cases": records,
                }
            )
        )
        (domain_dir / "score.json").write_bytes(canonical(scored))
        receipt["domains"][domain] = {
            key: scored[key]
            for key in ("scenes", "action_valid", "start_hit", "end_hit", "task_success",
                        "task_success_rate", "degenerate_start_equals_end")
        }
        print("H61_OFFICIAL_SCORE", domain, json.dumps(receipt["domains"][domain],
                                                       sort_keys=True), flush=True)
    (work / "official-eval-run.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--domains", default="slider,sheet,slide_resize,text_highlight")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--adapter-sha256", default=None)
    args = parser.parse_args()
    result = run(
        [item for item in args.domains.split(",") if item],
        args.work,
        limit=args.limit,
        adapter=args.adapter,
        adapter_sha256=args.adapter_sha256,
    )
    print(json.dumps(result["domains"], sort_keys=True))

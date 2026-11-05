"""H426 full image-processor supervision contract, CPU-only research preflight.

Validates the real H425 two-pass, image-expanded supervised-token boundary.
A passing synthetic test never proves a trained H426, source rights, or OOD.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = "Qwen/Qwen3-VL-2B-Instruct"
REV = "89644892e4d85e24eaac8bacfd4f463576704203"
JSON_SHA = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
CONFIG_SHA = "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399"


def _loss():
    path = Path(__file__).with_name("target_aware_loss.py")
    spec = importlib.util.spec_from_file_location("h426_processor_loss", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.source_coordinate_token_weights


def _ids(batch, key):
    if key not in batch:
        raise ValueError("H426 image processor missing " + key)
    raw = batch[key]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if (not isinstance(raw, list) or len(raw) != 1
        or not isinstance(raw[0], list)
        or any(type(i) is not int for i in raw[0])):
        raise ValueError("H426 expected one image-conditioned token batch")
    return raw[0]


def check(processor, image, request: str, answer: str, domain: str):
    """Reproduce H425 original head/full processor calls; return aggregates only.

    Do not include any original cases, predictions, images or labels in output.
    """
    if not request or len(request) > 4096 or not isinstance(answer, str):
        raise ValueError("H426 synthetic instruction/answer format invalid")
    user = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": request}]}]
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    head = processor.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        user + [assistant], tokenize=False, add_generation_prompt=False)
    original = processor(text=[head], images=[image], return_tensors="pt")
    completed = processor(text=[full], images=[image], return_tensors="pt")
    a = _ids(original, "input_ids")
    b = _ids(completed, "input_ids")
    attention = _ids(completed, "attention_mask")
    if (not a or len(b) <= len(a) or b[:len(a)] != a
        or len(attention) != len(b) or any(i != 1 for i in attention)):
        raise ValueError("H426 REAL image-patch-expanded supervised prefix drift")
    for batch in (original, completed):
        if "pixel_values" not in batch or "image_grid_thw" not in batch:
            raise ValueError("H426 REAL image patch/grid absent from processor")
    tokenizer = processor.tokenizer
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if (type(image_token) is not int or image_token < 0
        or image_token not in a or image_token in b[len(a):]):
        raise ValueError("H426 image patch tokens absent or supervised as answer")
    supervised = b[len(a):]
    pieces = [tokenizer.decode([i], skip_special_tokens=False)
              for i in supervised]
    if answer not in "".join(pieces):
        raise ValueError("H426 genuine image processor supervised answer changed")
    weights = _loss()(pieces, domain)
    if len(weights) != len(supervised) or sum(x > 1 for x in weights) < 2:
        raise ValueError("H426 supervised start/end numeric field token loss invalid")
    return {
        "schema": "hala.h426.full-image-processor-boundary.v1",
        "image_patch_tokens": a.count(image_token),
        "source_prefix_tokens": len(a),
        "supervised_tokens": len(supervised),
        "weighted_coordinate_tokens": sum(x > 1 for x in weights),
        "cpu_research_only": True,
        "new_model_trained": False,
        "independent_external_ood": False,
        "model_release_qualified": False,
    }


def run(snapshot: Path):
    """Run synthetic full Qwen3-VL processor only; never fetch model weights."""
    from PIL import Image
    from transformers import AutoProcessor
    root = Path(snapshot).resolve()
    if root.name != REV:
        raise ValueError("H426 immutable Qwen3-VL processor revision mismatch")
    for name, expected in (("tokenizer.json", JSON_SHA),
                           ("tokenizer_config.json", CONFIG_SHA)):
        if not (root/name).is_file():
            raise ValueError("H426 missing pinned tokenizer file " + name)
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError("H426 pinned tokenizer file digest drift: " + name)
    processor = AutoProcessor.from_pretrained(
        str(root), trust_remote_code=False, local_files_only=True, use_fast=False)
    img = Image.new("RGB", (256, 256), (236, 237, 238))
    targets = {
        "slider": [[300, 400], [700, 400]],
        "slide_resize": [[100, 100], [900, 900]],
        "text_highlight": [[120, 420], [660, 420]],
    }
    results = {}
    for domain, (start, end) in targets.items():
        answer = json.dumps({"type": "drag", "start_point": start,
            "end_point": end, "ordered": True,
            "coordinate_space": "normalized_1000"}, separators=(",", ":"))
        results[domain] = check(processor, img,
            "On this synthetic image, describe the drag; return JSON only. Do not act.",
            answer, domain)
    return {
        "model": BASE, "revision": REV,
        "synthetic_image_only": True, "no_model_weights_loaded": True,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().tokenizer_snapshot), sort_keys=True))

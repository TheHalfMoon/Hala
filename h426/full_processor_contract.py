"""H426 full image-processor supervision contract, CPU-only research preflight.

Validates the real H425 two-pass, image-expanded supervised-token boundary.
A passing synthetic test never proves a trained H426, source rights, or OOD.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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
        pixels, grid = batch["pixel_values"], batch["image_grid_thw"]
        if pixels is None or grid is None:
            raise ValueError("H426 REAL image patch/grid empty")
        if hasattr(pixels, "numel") and pixels.numel() == 0:
            raise ValueError("H426 REAL image patch/grid empty")
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if (not isinstance(grid, list) or len(grid) != 1
            or not isinstance(grid[0], list) or len(grid[0]) != 3
            or any(type(v) is not int or v <= 0 for v in grid[0])):
            raise ValueError("H426 REAL image patch/grid malformed")
    # H216 already established that identical visual grid metadata is NOT
    # sufficient: the real pixel tensors can differ, or contain NaN/Infinity.
    # Refuse supervision unless model-visible pixel payloads are identical
    # and finite in both the H425 header and completed-answer passes.
    before_pixels = original["pixel_values"]
    after_pixels = completed["pixel_values"]
    if callable(getattr(before_pixels, "numel", None)):
        if not callable(getattr(after_pixels, "numel", None)):
            raise ValueError("H426 visual tensor representation drift")
        import torch
        if (before_pixels.numel() <= 0
            or before_pixels.numel() != after_pixels.numel()
            or not torch.equal(before_pixels, after_pixels)
            or not bool(torch.isfinite(before_pixels).all().item())):
            raise ValueError("H426 image processor pixels unequal or nonfinite")
    elif type(before_pixels) is list and type(after_pixels) is list:
        # Explicit synthetic fake-processor pathway for source-only CI.
        # Real AutoProcessor(..., return_tensors='pt') supplies tensors.
        def finite_nested(values):
            if type(values) is list:
                return bool(values) and all(finite_nested(v) for v in values)
            return type(values) in (int, float) and math.isfinite(values)
        if (before_pixels != after_pixels
            or not finite_nested(before_pixels)
            or not finite_nested(after_pixels)):
            raise ValueError("H426 image processor pixels unequal or nonfinite")
    else:
        raise ValueError("H426 image processor unexpected pixel representation")
    def grid_values(batch):
        grid = batch["image_grid_thw"]
        return grid.tolist() if hasattr(grid, "tolist") else grid
    if grid_values(original) != grid_values(completed):
        raise ValueError("H426 REAL image grid drift between head and full")
    tokenizer = processor.tokenizer
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if (type(image_token) is not int or image_token < 0
        or image_token not in a or image_token in b[len(a):]):
        raise ValueError("H426 image patch tokens absent or supervised as answer")
    # Official Qwen3-VL processor expands each image into precisely
    # prod(image_grid_thw) // (image_processor.merge_size ** 2) image_pad IDs.
    # A merely present image marker is insufficient evidence of patch alignment.
    merge_size=getattr(getattr(processor, "image_processor", None), "merge_size", None)
    if type(merge_size) is not int or merge_size<=0:
        raise ValueError("H426 actual Qwen image processor merge size missing")
    grid=grid_values(completed)[0]
    patches=grid[0]*grid[1]*grid[2]
    merged=merge_size*merge_size
    if (patches<merged or patches%merged or
        a.count(image_token)!=patches//merged or
        b.count(image_token)!=patches//merged):
        raise ValueError("H426 expanded image patch count mismatches Qwen merge grid")
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
    """Offline actual Qwen3-VL full-processor preflight on synthetic pixels.

    Uses H218/H387's existing *eight-file* provenance/copy isolation. A mere
    tokenizer-only hash never authorizes loading processor config from disk.
    No image source dataset, base model weights, LoRA or GPU is loaded.
    """
    if (os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("H426 full processor must run with offline flags")
    root = Path(snapshot)
    if not root.is_absolute() or root.name != REV or root.is_symlink():
        raise ValueError("H426 full processor requires exact absolute revision")
    # Gate all eight preprocessor/tokenizer/chat/video/source files,
    # including configs beyond the two historical H426 tokenizer SHA pins.
    from h218.pinned_qwen_processor_bytes import (
        PINNED_PUBLIC_FILES, verify_pinned_snapshot,
    )
    expected=verify_pinned_snapshot(root)
    if (expected.get("verified_files") != 8
        or len(PINNED_PUBLIC_FILES) != 8
        or PINNED_PUBLIC_FILES["tokenizer.json"][1] != JSON_SHA
        or PINNED_PUBLIC_FILES["tokenizer_config.json"][1] != CONFIG_SHA):
        raise ValueError("H426 entire Qwen processor eight-file source pin drift")
    from h387.verified_processor_handoff import verified_processor_handoff
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLProcessor, __version__
    if __version__ != "4.57.3":
        raise ValueError("H426 processor requires original H425 Transformers 4.57.3")
    # Keep the *verified staged copy* alive throughout processor construction
    # and all synthetic image passes. Never reopen mutable owner cache files.
    with verified_processor_handoff(root) as pinned_copy:
        processor = AutoProcessor.from_pretrained(
            str(pinned_copy), trust_remote_code=False,
            local_files_only=True, use_fast=False,
        )
        if type(processor) is not Qwen3VLProcessor:
            raise ValueError("H426 actual pinned Qwen3VLProcessor class required")
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
        "verified_processor_files": 8,
        "transformers_version": "4.57.3",
        "synthetic_image_only": True, "no_model_weights_loaded": True,
        "training_executed": False, "external_ood": False,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().tokenizer_snapshot), sort_keys=True))

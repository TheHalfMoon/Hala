"""H209 CPU-only public pretrained VLM baseline on two self-generated colors.

Owner-only research run. Never opens owner media, old OOD, current
quarantined screenshots, gold, any browser or application DOM.
Uses a public Apache-2.0 256M baseline, not Hala trained weights.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
MODEL_REVISION = "9b299836fe84e0aabd39e1e07651e569e5dfe87c"
COLORS = ("RED", "BLUE")
MIN_FREE_GIB = 3


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H209 real public baseline: " + reason)


def synthetic_image(color: str):
    from PIL import Image, ImageDraw
    require(color in COLORS, "invalid self-generated public color")
    img = Image.new("RGB", (960, 720), "white")
    ImageDraw.Draw(img).rectangle((250, 160, 710, 560), fill=color.lower())
    return img


def generated_continuation(model, processor, img, task, max_new_tokens=12):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": task},
    ]}]
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cpu")
    require("pixel_values" in batch and batch["pixel_values"].numel() > 0,
            "missing actual vision tensor")
    prompt_ids = batch["input_ids"][0].tolist()
    require(len(prompt_ids) > 0, "missing prompt token sequence")
    with torch.inference_mode():
        result = model.generate(
            **batch, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
        )
    require(len(result) == 1, "expected exactly one model output")
    all_tokens = result[0].tolist()
    require(len(all_tokens) > len(prompt_ids)
            and all_tokens[:len(prompt_ids)] == prompt_ids,
            "invalid model prompt prefix or no continuation")
    answer = processor.batch_decode(
        [all_tokens[len(prompt_ids):]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    require(len(answer) == 1 and isinstance(answer[0], str),
            "invalid decoded continuation")
    return answer[0], len(all_tokens)-len(prompt_ids), int(batch["pixel_values"].numel())


def main():
    parser = argparse.ArgumentParser(description="H209 local PUBLIC 256M pretrained VLM baseline only")
    parser.add_argument("--temporary-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.temporary_root
    require(root.is_absolute() and root.name == "Hala-H209-QA",
            "only H209 disposable test environment is permitted")
    require(shutil.disk_usage(root).free >= MIN_FREE_GIB*1024**3,
            "insufficient free disk for public model smoke")
    require(os.environ.get("HF_HOME") == str(root / "hf"),
            "model cache must be in isolated disposable H209 QA directory")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float32, trust_remote_code=False,
    ).to("cpu")
    model.eval()
    print("H209_REAL_PUBLIC_CPU_WEIGHTS_AND_PROCESSOR_LOADED revision=" + MODEL_REVISION,
          flush=True)
    results = []
    for color in COLORS:
        task = "What color is the large central filled rectangle? Answer RED or BLUE, exactly one word."
        answer, tokens, visual_elements = generated_continuation(
            model, processor, synthetic_image(color), task, max_new_tokens=12)
        # Do not publish raw unconstrained model text or any user content.
        normalized = answer.strip().upper()
        recognized = normalized in COLORS
        results.append({
            "target": color,
            "recognized_one_word": recognized,
            "correct": recognized and normalized == color,
            "new_tokens": tokens,
            "real_vision_tensor_elements": visual_elements,
        })
        print("H209_REAL_PUBLIC_CPU_COLOR_PROMPT",
              "target=" + color,
              "recognized=" + str(recognized),
              "correct=" + str(bool(recognized and normalized == color)),
              "generated_tokens=" + str(tokens),
              "visual_elements=" + str(visual_elements),
              flush=True)
    # Never score these as independent unseen GUI benchmark cases.
    print("H209_PUBLIC_PRETRAINED_REAL_INFERENCE_FINISHED",
          "model_calls=2",
          "self_generated_color_answers_correct=" + str(sum(int(x["correct"]) for x in results)),
          "new_hala_training_steps=0",
          "official_ood_calls=0",
          "private_media_opened=0",
          "no_gui_action_claim=true",
          "elapsed_seconds=" + str(round(time.monotonic()-start,1)),
          flush=True)


if __name__ == "__main__":
    main()

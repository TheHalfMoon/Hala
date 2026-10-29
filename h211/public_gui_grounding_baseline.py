"""H211: real *public pretrained* CPU VLM on original self-authored GUI pixels.

Never Hala fine-tuning, held-out OOD, owner media/gold or original private
source. Measures direct strict native JSON GUI click grounding in-memory
without dispatching real mouse/keyboard input or reading any DOM.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from h199.native_action_sft import parse_actions
from h209.public_cpu_visual_baseline import (
    MIN_FREE_GIB, MODEL_ID, MODEL_REVISION, generated_continuation, require,
)

SCENES = {
    "LEFT": (80, 250, 340, 380),
    "RIGHT": (620, 250, 880, 380),
}
VIEWPORT = (960, 720)
MODEL_TASK = (
    "See this screenshot of a computer application. "
    "Click the visible blue SAVE button. "
    "Answer with ONLY one JSON object exactly like "
    '{"actions":[{"kind":"click","point":[x,y]}]}. '
    "Use pixel x 0-959 and y 0-719. Do not describe the picture."
)
MAX_NEW_TOKENS = 64


def original_synthetic_gui(side: str):
    """Make a fresh self-authored image, never read any external pixels."""
    from PIL import Image, ImageDraw, ImageFont
    require(type(side) is str and side in SCENES, "unknown synthetic GUI scene")
    image = Image.new("RGB", VIEWPORT, "#f5f6f8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 959, 75), fill="#1c2330")
    draw.text((42, 37), "Research demo", fill="white", anchor="lm")
    draw.text((480, 155), "Save your draft", fill="#1c2330", anchor="mm")
    save = SCENES[side]
    other = SCENES["RIGHT" if side == "LEFT" else "LEFT"]
    draw.rounded_rectangle(save, radius=18, fill="#1c61d5")
    draw.rounded_rectangle(other, radius=18, fill="#9da4ad")
    # Large buttons for a small 256M public baseline.
    font = ImageFont.load_default(size=36)
    draw.text(((save[0]+save[2])//2, (save[1]+save[3])//2),
              "SAVE", fill="white", anchor="mm", font=font)
    draw.text(((other[0]+other[2])//2, (other[1]+other[3])//2),
              "CANCEL", fill="white", anchor="mm", font=font)
    return image


def grade_native_gui_action(raw: str, side: str) -> str:
    """Strict action grammar; no fuzzy text scoring or invented clicks."""
    require(type(side) is str and side in SCENES,
            "unregistered source panel")
    try:
        action = parse_actions(raw)["actions"][0]
    except (ValueError, TypeError, KeyError, IndexError):
        return "MALFORMED"
    if action["kind"] == "abstain":
        return "ABSTAIN"
    if action["kind"] != "click":
        return "NONCLICK"
    point = action["point"]
    left, top, right, bottom = SCENES[side]
    if left <= point[0] <= right and top <= point[1] <= bottom:
        return "HIT"
    return "MISS"


def execute_public_model_baseline(temp: Path) -> dict:
    """Explicit owner-local only; model cache temporary and private-safe."""
    require(isinstance(temp, Path) and temp.is_absolute()
            and temp.name == "Hala-H211-QA", "only isolated H211 QA allowed")
    require(temp.is_dir() and shutil.disk_usage(temp).free >=
            MIN_FREE_GIB*1024**3,
            "insufficient safe free space for public 256M model")
    require(os.environ.get("HF_HOME") == str(temp / "hf"),
            "model may cache only in H211 isolated directory")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        dtype=torch.float32, trust_remote_code=False).to("cpu")
    model.eval()
    print("H211_REAL_PUBLIC_256M_CPU_MODEL_LOADED", flush=True)
    observations = []
    for side in SCENES:
        raw, new_tokens, pixel_elements = generated_continuation(
            model, processor, original_synthetic_gui(side),
            MODEL_TASK, max_new_tokens=MAX_NEW_TOKENS)
        grade = grade_native_gui_action(raw, side)
        observations.append(grade)
        # NEVER print raw model text, private SHA, or screenshot. In fact
        # there are no private inputs in this original synthetic experiment.
        print("H211_ACTUAL_PRETRAINED_GUI_GROUNDING",
              "panel="+side, "outcome="+grade,
              "generated_tokens="+str(new_tokens),
              "actual_vision_tensor_elements="+str(pixel_elements),
              flush=True)
    print("H211_PUBLIC_REAL_GUI_CLICK_DIAGNOSTIC_COMPLETE",
          "model_calls="+str(len(SCENES)),
          "hits="+str(observations.count("HIT")),
          "misses="+str(observations.count("MISS")),
          "malformed="+str(observations.count("MALFORMED")),
          "nonclick="+str(observations.count("NONCLICK")),
          "abstain="+str(observations.count("ABSTAIN")),
          "native_browser_events=0",
          "hala_optimizer_steps=0",
          "official_ood_cases=0",
          "private_media_opened=0",
          "elapsed_seconds="+str(round(time.monotonic()-start,1)),
          flush=True)
    return {"diagnostic_observations": observations,
            "real_model_calls": len(SCENES),
            "trained_hala": False, "official_ood_evaluated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporary-root", type=Path, required=True)
    args = parser.parse_args()
    execute_public_model_baseline(args.temporary_root)


if __name__ == "__main__":
    main()

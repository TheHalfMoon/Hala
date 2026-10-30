"""H239: offline real Qwen3-VL processor parity for bilingual type_text.

Self-authored synthetic screenshot ONLY, no owner images, no H173 OOD,
no model weights, no model.generate, no optimizer or TRAIN admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, assistant_completion_mask,
)
from h216.real_qwen_processor_alignment import (
    MAX_FULL_TOKENS, MAX_PREFIX_TOKENS, QWEN_PROCESSOR_REV,
    token_ids, verify_cached_processor_root, visual_tensor_receipt,
)
from h234.native_type_text import (
    parse_actions_v4, synthetic_sft_messages_v4,
)

SAMPLES = (
    ("en", "Enter the synthetic test value in the visible input.", "Demo-123"),
    ("ar", "اكتب النص التجريبي في حقل الإدخال الظاهر.", "مرحبا ١٢٣"),
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H239 bilingual processor gate: " + reason)


def make_synthetic_image():
    """PIL in-memory invented image; no historical or owner screenshot load."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (960, 720), "#f1f2f4")
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 85, 870, 540), fill="#ffffff", outline="#262626", width=3)
    draw.rectangle((110, 155, 665, 205), fill="#ffffff",
                   outline="#425979", width=3)
    draw.rectangle((110, 255, 225, 300), fill="#bdcce1", outline="#334455")
    return img


def audit_type_text_sample(processor, *, language: str, task: str,
                           text: str, image=None) -> dict:
    """Actual apply_chat_template parity, processor-only and no model call."""
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "official original source TRAIN manifest has not been admitted")
    require(language in ("en", "ar"), "only two explicit research languages")
    require(type(task) is str and type(text) is str,
            "synthetic input task/text required")
    if image is None:
        image = make_synthetic_image()
    require(image.size == (960, 720) and image.mode == "RGB",
            "expected one self-authored 960x720 RGB screenshot")
    source_json = json.dumps({
        "actions": [{"kind": "type_text", "point": [185, 176], "text": text}],
    }, ensure_ascii=False, separators=(",", ":"))
    native = parse_actions_v4(source_json, training_target=True)
    question, answer = synthetic_sft_messages_v4(task, source_json)
    question[0]["content"][0]["image"] = image
    # The answer must use the IDENTICAL source PIL object, never reload media.
    answer[0]["content"][0]["image"] = image
    require(question[0]["content"][0]["image"] is answer[0]["content"][0]["image"],
            "different source images for prompt and completion")
    require("type_text" in question[0]["content"][1]["text"],
            "model instruction omitted text action vocabulary")
    require(text not in question[0]["content"][1]["text"],
            "reference completion text leaked into visible task prompt")

    opts = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
    prompt = processor.apply_chat_template(
        question, add_generation_prompt=True, **opts)
    full = processor.apply_chat_template(
        answer, add_generation_prompt=False, **opts)
    before = token_ids(prompt, what="type_text prefix")
    after = token_ids(full, what="type_text completion")
    require(0 < len(before) <= MAX_PREFIX_TOKENS
            and len(before) + 5 <= len(after) <= MAX_FULL_TOKENS,
            "unsafe or missing multimodal token sequence")
    elements = visual_tensor_receipt(prompt, full)
    require(after[:len(before)] == before,
            "prompt prefix differs after adding bilingual assistant completion")
    mask = assistant_completion_mask(before, after)
    require(mask[:len(before)] == [-100] * len(before)
            and mask[len(before):] == after[len(before):],
            "loss mask includes source prompt or visual prefix")
    decoded = processor.batch_decode(
        [after[len(before):]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    require(type(decoded) is list and len(decoded) == 1
            and type(decoded[0]) is str
            and parse_actions_v4(decoded[0], training_target=True) == native,
            "bilingual assistant type_text did not survive actual tokenizer")
    return {
        "language": language,
        "status": "H239_PROCESSOR_ONLY_TYPE_TEXT_NOT_DATA_ADMISSION",
        "input_prefix_exact": True,
        "assistant_completion_only_mask": True,
        "text_action_roundtrip": True,
        "vision_identical": True,
        "prompt_tokens": len(before),
        "completion_tokens": len(after) - len(before),
        "vision_elements": elements,
        "model_loaded": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "training_approved": False,
        "ood_evaluation_approved": False,
    }


def run_offline_real_processor(snapshot: Path) -> list[dict]:
    """Optional owner Windows only: immutable pinned local Qwen processor."""
    require(os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "network disabled flags required before importing processor")
    verify_cached_processor_root(snapshot)
    from transformers import AutoProcessor, Qwen3VLProcessor
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    require(type(processor) is Qwen3VLProcessor,
            "unexpected tokenizer / image processor class")
    output = []
    for lang, task, target in SAMPLES:
        result = audit_type_text_sample(
            processor, language=lang, task=task, text=target)
        output.append(result)
    require(len(output) == 2
            and {r["language"] for r in output} == {"en", "ar"},
            "both bilingual processor actions mandatory")
    return output


def main() -> None:
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--local-processor-snapshot", type=Path, required=True)
    ns = arg.parse_args()
    try:
        output = run_offline_real_processor(ns.local_processor_snapshot)
    except Exception:
        # Never print raw model-visible text, private paths, or processor internals.
        raise SystemExit("H239_BILINGUAL_REAL_PROCESSOR_FAIL_CLOSED") from None
    print("H239_REAL_QWEN3VL_BILINGUAL_TYPE_TEXT_PROCESSOR_PASS",
          "examples=2", "languages=en,ar", "model_calls=0",
          "optimizer_steps=0", "train=0", "dev=0", "ood=0", flush=True)


if __name__ == "__main__":
    main()

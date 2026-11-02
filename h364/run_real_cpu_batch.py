"""Opt-in REAL PINNED CPU Qwen3VL processor native multimodal batch QA.

Public authored source-only candidates, no training admission, weights, loss,
model inference or GPU. Original screenshots/gold kept ONLY in TEMP.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, parse_actions,
    source_task_prompt, training_messages_for_synthetic_contract,
)
from h216.real_qwen_processor_alignment import (
    token_ids, verify_cached_processor_root, visual_tensor_receipt,
)
from h357.source_candidate import capture_source_candidate
from h358.source_candidate import capture_candidate
from h359.native_candidate_bridge import PREFIX, audit_review_bundle, safe_regular
from h361.original_qwen_processor import _candidate_rows
from h364.native_batch import SCHEMA, ROLE, real_cpu_visual_batch


def require(ok: bool):
    if not ok:
        raise ValueError("H364_PINNED_REAL_CPU_SFT_BATCH_PREFLIGHT_BLOCKED")


def verify_one(processor, root: Path, family: str, index: int) -> tuple[dict, str]:
    from PIL import Image
    row = _candidate_rows(root, family)[index]  # audits ALL original rows
    require(row["prompt"].startswith(PREFIX))
    task = row["prompt"][len(PREFIX):]
    require(source_task_prompt(task) == row["prompt"])
    native = parse_actions(row["completion"], training_target=True)
    media = safe_regular(root, row["image_path"]).read_bytes()
    require(sha256(media).hexdigest() == row["image_sha256"])
    with Image.open(BytesIO(media)) as original:
        require(original.format == "PNG" and original.size == (960, 720))
        image = original.convert("RGB")
    question, answer = training_messages_for_synthetic_contract(
        task, row["completion"])
    question[0]["content"][0]["image"] = image
    answer[0]["content"][0]["image"] = image
    params = dict(tokenize=True, return_dict=True, return_tensors="pt")
    before = processor.apply_chat_template(
        question, add_generation_prompt=True, **params)
    after = processor.apply_chat_template(
        answer, add_generation_prompt=False, **params)
    original_prefix = token_ids(before, what="H364 original source prompt")
    full = token_ids(after, what="H364 original source completion")
    require(full[:len(original_prefix)] == original_prefix)
    visual_tensor_receipt(before, after)
    decoded = processor.batch_decode(
        [full[len(original_prefix):]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    require(len(decoded) == 1 and
            parse_actions(decoded[0], training_target=True) == native)
    return {
        "prefix_ids": original_prefix, "full_ids": full,
        "pixel_values": after["pixel_values"],
        "image_grid_thw": after["image_grid_thw"],
    }, native["actions"][0]["kind"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-processor-snapshot", required=True, type=Path)
    parser.add_argument("--browser", required=True, choices=("msedge", "chromium"))
    args = parser.parse_args()
    require(not ADMITTED_TRAIN_MANIFEST_SHAS and
            os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1" and
            os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1")
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_available())
    verify_cached_processor_root(args.local_processor_snapshot)
    from transformers import AutoProcessor, Qwen3VLProcessor
    processor = AutoProcessor.from_pretrained(
        str(args.local_processor_snapshot),
        local_files_only=True, trust_remote_code=False)
    require(type(processor) is Qwen3VLProcessor)
    pad = processor.tokenizer.pad_token_id
    require(type(pad) is int and pad >= 0)

    with TemporaryDirectory(prefix="hala-h364-public-source-only-") as folder:
        root = Path(folder)
        a, b = root / "controlgarden", root / "noterelay"
        capture_source_candidate(a, browser_kind=args.browser)
        capture_candidate(b, browser_kind=args.browser)
        # Independently re-audit ALL 192 rows and original image SHA bytes.
        for family, home in (
            ("controlgarden-single-app", a),
            ("noterelay-original-multistep", b),
        ):
            audit = audit_review_bundle(home, family)
            require(audit["official_train_admitted"] is False and
                    audit["can_train"] is False)
        first, k1 = verify_one(processor, a, "controlgarden-single-app", 0)
        second, k2 = verify_one(processor, b, "noterelay-original-multistep", 6)
        require((k1, k2) == ("click", "type_text"))
        batch = real_cpu_visual_batch([first, second], pad)
        supervised = int((batch["labels"] != -100).sum().item())
        padded = int((batch["attention_mask"] == 0).sum().item())
        require(supervised > 0 and padded >= 0 and
                tuple(batch["image_grid_thw"].shape) == (2, 3))
        receipt = {
            "schema": SCHEMA, "role": ROLE,
            "actual_processor": "Qwen3VLProcessor",
            "processor_revision": args.local_processor_snapshot.name,
            "review_only_original_source_rows": 2,
            "family_count": 2,
            "actions": {"click": 1, "type_text": 1},
            "languages": {"en": 1, "ar": 1},
            "all_192_source_rows_audited": True,
            "visual_patches": int(batch["pixel_values"].shape[0]),
            "image_grids": int(batch["image_grid_thw"].shape[0]),
            "padded_tokens": padded,
            "assistant_supervised_tokens": supervised,
            "batch_tensor_dimensions_valid": True,
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "optimizer_steps": 0,
            "official_train_admitted": False,
            "new_blind_ood": False,
            "can_train": False, "can_release": False,
        }
        del batch, first, second
    # Gold, source screenshots and image arrays deleted before receipt output.
    print("H364_REAL_CPU_NATIVE_BATCH_ONLY",
          json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

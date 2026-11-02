"""H367: four-row original-source pinned Qwen3-VL CPU packing evidence.

Source-only review candidates, NOT lawful TRAIN or blind OOD.
No model weights, loss, forward, optimizer, credentials, GPU or paid service.
Only aggregate booleans and counts leave the owner-local ephemeral directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
from h216.real_qwen_processor_alignment import verify_cached_processor_root
from h357.source_candidate import candidate_specs, capture_source_candidate
from h358.source_candidate import scenes, capture_candidate
from h359.native_candidate_bridge import audit_review_bundle
from h361.original_qwen_processor import _candidate_rows
from h364.native_batch import real_cpu_visual_batch
from h364.run_real_cpu_batch import verify_one

SCHEMA = "hala.h367.four-original-source-qwen-cpu-preflight.v1"
# Each language has both original single-click and multistep type_text.
# These are previously EXPOSED public candidate indexes, not new OOD.
SELECTION = (
    ("controlgarden-single-app", 0, "en", "click"),
    ("controlgarden-single-app", 1, "ar", "click"),
    ("noterelay-original-multistep", 2, "en", "type_text"),
    ("noterelay-original-multistep", 6, "ar", "type_text"),
)


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H367_FOUR_ROW_PROCESSOR_ONLY_REFUSED")


def source_selection() -> tuple[tuple[str, int, str, str], ...]:
    """Pure source policy check; doesn't generate screenshots or model calls."""
    require(not ADMITTED_TRAIN_MANIFEST_SHAS and len(SELECTION) == 4)
    controls = candidate_specs()
    notes = scenes()
    require(len(controls) == 64 and len(notes) == 32
            and len({(family, index) for family, index, _, _ in SELECTION}) == 4)
    for family, index, language, action in SELECTION:
        require(type(index) is int and type(language) is str)
        if family == "controlgarden-single-app":
            require(controls[index]["language"] == language and action == "click")
        else:
            require(family == "noterelay-original-multistep"
                    and notes[index // 4]["language"] == language
                    and ("open", "focus", "type", "save")[index % 4] == "type"
                    and action == "type_text")
    require(Counter(lang for _, _, lang, _ in SELECTION) ==
            {"en": 2, "ar": 2}
            and Counter(kind for _, _, _, kind in SELECTION) ==
            {"click": 2, "type_text": 2})
    return SELECTION


def verify_pack_order(batch: dict, examples: list[dict]) -> dict:
    """Reject accidental token/visual-grid/visual-patch reordering.

    Call on both normal and reversed original sample sequences. The
    preflight accepts a specific verified source mapping only; do not
    reinterpret this as independently tamper-proof provenance or TRAIN.
    """
    import torch
    require(type(batch) is dict and type(examples) is list and
            len(examples) == 4 and tuple(batch["input_ids"].shape)[0] == 4)
    patch_offset = 0
    supervised = 0
    pad = 0
    for i, item in enumerate(examples):
        full = item["full_ids"]
        prefix = item["prefix_ids"]
        patch_rows = item["pixel_values"].shape[0]
        row_input = batch["input_ids"][i].tolist()
        row_label = batch["labels"][i].tolist()
        row_attention = batch["attention_mask"][i].tolist()
        require(row_input[:len(full)] == full
                and row_label[:len(prefix)] == [-100] * len(prefix)
                and row_label[len(prefix):len(full)] == full[len(prefix):]
                and row_label[len(full):] == [-100] * (len(row_label) - len(full))
                and row_attention == [1] * len(full) +
                [0] * (len(row_attention) - len(full))
                and torch.equal(batch["image_grid_thw"][i], item["image_grid_thw"][0])
                and torch.equal(
                    batch["pixel_values"][patch_offset:patch_offset + patch_rows],
                    item["pixel_values"]))
        patch_offset += patch_rows
        supervised += len(full) - len(prefix)
        pad += len(row_input) - len(full)
    require(patch_offset == batch["pixel_values"].shape[0]
            and supervised == int((batch["labels"] != -100).sum().item())
            and pad == int((batch["attention_mask"] == 0).sum().item()))
    return {"visual_patch_rows": patch_offset,
            "assistant_supervised_tokens": supervised,
            "masked_pad_tokens": pad}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-processor-snapshot", required=True, type=Path)
    parser.add_argument("--browser", required=True, choices=("msedge", "chromium"))
    args = parser.parse_args()
    require(not ADMITTED_TRAIN_MANIFEST_SHAS
            and os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            and os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1")
    selection = source_selection()
    # Verify exact SHA of eight pinned PUBLIC processor assets BEFORE imports.
    verify_cached_processor_root(args.local_processor_snapshot)
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_available())
    from transformers import AutoProcessor, Qwen3VLProcessor
    processor = AutoProcessor.from_pretrained(
        str(args.local_processor_snapshot), local_files_only=True,
        trust_remote_code=False)
    require(type(processor) is Qwen3VLProcessor)
    pad_id = processor.tokenizer.pad_token_id
    require(type(pad_id) is int and pad_id >= 0)
    with TemporaryDirectory(prefix="hala-h367-public-source-only-") as folder:
        root = Path(folder)
        sources = {
            "controlgarden-single-app": root / "controlgarden",
            "noterelay-original-multistep": root / "noterelay",
        }
        capture_source_candidate(sources["controlgarden-single-app"],
                                 browser_kind=args.browser)
        capture_candidate(sources["noterelay-original-multistep"],
                          browser_kind=args.browser)
        audits = [
            audit_review_bundle(sources[family], family)
            for family in sources
        ]
        require(len(audits) == 2 and all(
            a["can_train"] is False and
            a["official_train_admitted"] is False for a in audits))
        examples = []
        with torch.no_grad():
            for family, index, lang, kind in selection:
                source_row = _candidate_rows(sources[family], family)[index]
                require(source_row["language"] == lang)
                if family == "noterelay-original-multistep":
                    require(source_row["phase"] == "type")
                example, actual_kind = verify_one(
                    processor, sources[family], family, index)
                require(actual_kind == kind)
                examples.append(example)
            batch = real_cpu_visual_batch(examples, pad_id)
            normal = verify_pack_order(batch, examples)
            reversed_samples = list(reversed(examples))
            swapped = real_cpu_visual_batch(reversed_samples, pad_id)
            reverse = verify_pack_order(swapped, reversed_samples)
            require(reverse == normal
                    and tuple(batch["image_grid_thw"].shape) == (4, 3)
                    and tuple(swapped["image_grid_thw"].shape) == (4, 3)
                    and batch["pixel_values"].shape[0] ==
                    swapped["pixel_values"].shape[0])
        receipt = {
            "schema": SCHEMA,
            "real_pinned_processor": "Qwen3VLProcessor",
            "four_original_candidate_rows": 4,
            "original_candidate_images_audited": 192,
            "language_counts": {"en": 2, "ar": 2},
            "action_counts": {"click": 2, "type_text": 2},
            "normal_and_reverse_order_exact": True,
            "four_image_grids": 4,
            **normal,
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "optimizer_steps": 0,
            "official_train_admitted": False,
            "blind_ood": False,
            "can_train": False,
            "can_release": False,
        }
        del batch, swapped, examples, reversed_samples
    print("H367_REAL_PINNED_CPU_FOUR_ROW_RESEARCH_ONLY",
          json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

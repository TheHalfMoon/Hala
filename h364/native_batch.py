"""H364: STRICT H199-native Qwen multimodal SFT batch *preflight*, never training.

This source-only research collator exercises token padding, visual feature
packing and assistant-only labels.  It never loads a Qwen MODEL, computes
loss, returns an optimizer, or admits H357/H358 to legal TRAIN.
"""
from __future__ import annotations

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, assistant_completion_mask,
)
from h216.real_qwen_processor_alignment import (
    MAX_FULL_TOKENS, MAX_PREFIX_TOKENS,
)

ROLE = "H364_RESEARCH_ONLY_CPU_BATCH_NOT_ADMITTED_TRAIN"
SCHEMA = "hala.h364.native-multimodal-batch-preflight.v1"
EXPECTED_KEYS = frozenset({"prefix_ids", "full_ids", "pixel_values", "image_grid_thw"})
MAX_ROWS = 4


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("H364_NON_TRAINING_NATIVE_BATCH_REJECTED")


def padded_token_plan(examples: list[dict], pad_id: int) -> dict:
    """Pure CPU lists, NO tensor import. All user+image tokens masked.

    Right padding is deliberately chosen: label=-100 and attention=0 on
    PAD, while assistant suffix is unchanged and always supervises one
    native H199 JSON action.  Never infer source permissions from data.
    """
    require(not ADMITTED_TRAIN_MANIFEST_SHAS
            and type(pad_id) is int and pad_id >= 0
            and type(examples) is list and 2 <= len(examples) <= MAX_ROWS)
    original_ids = []
    masks = []
    for sample in examples:
        require(type(sample) is dict
                and {"prefix_ids", "full_ids"}.issubset(sample)
                and type(sample["prefix_ids"]) is list
                and type(sample["full_ids"]) is list)
        prefix, full = sample["prefix_ids"], sample["full_ids"]
        require(all(type(value) is int and value >= 0 for value in prefix + full)
                and 0 < len(prefix) <= MAX_PREFIX_TOKENS
                and len(prefix) + 5 <= len(full) <= MAX_FULL_TOKENS
                and full[:len(prefix)] == prefix)
        expected = assistant_completion_mask(prefix, full)
        require(expected[:len(prefix)] == [-100] * len(prefix)
                and expected[len(prefix):] == full[len(prefix):])
        original_ids.append(full)
        masks.append(expected)
    longest = max(len(item) for item in original_ids)
    inputs, labels, attention = [], [], []
    for full, mask in zip(original_ids, masks):
        pad = longest - len(full)
        inputs.append(full + [pad_id] * pad)
        labels.append(mask + [-100] * pad)
        attention.append([1] * len(full) + [0] * pad)
    for full, row_inputs, row_labels, row_attention, mask in zip(
            original_ids, inputs, labels, attention, masks):
        require(len(row_inputs) == len(row_labels) ==
                len(row_attention) == longest
                and row_inputs[:len(full)] == full
                and row_labels[:len(mask)] == mask
                and row_attention == [1] * len(full) +
                [0] * (longest-len(full))
                and all(x == -100 for x in
                        row_labels[len(full):]))
    require(sum(sum(value != -100 for value in row)
                for row in labels) > 0)
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": attention,
        "sequence_lengths": [len(x) for x in original_ids],
        "max_length": longest,
        "supervised_token_counts": [
            sum(value != -100 for value in row) for row in masks
        ],
    }


def real_cpu_visual_batch(examples: list[dict], pad_id: int) -> dict:
    """Pure CPU zero-gradient research batch from verified processor outputs.

    No permanent artifact and no TRAIN permissions. Images are cat along
    the Qwen multimodal patch axis, while image_grid_thw retains 1 image
    grid per sample in the SAME row order as the token sequences.
    """
    require(not ADMITTED_TRAIN_MANIFEST_SHAS
            and type(examples) is list and 2 <= len(examples) <= MAX_ROWS
            and all(type(x) is dict and set(x) == EXPECTED_KEYS
                    for x in examples))
    from h216.real_qwen_processor_alignment import visual_tensor_receipt
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_available())
    plan = padded_token_plan(examples, pad_id)
    pixels, grids = [], []
    feature_width = None
    for sample in examples:
        raw_pixels = sample["pixel_values"]
        grid = sample["image_grid_thw"]
        require(type(raw_pixels) is torch.Tensor
                and type(grid) is torch.Tensor
                and raw_pixels.device.type == grid.device.type == "cpu"
                and raw_pixels.ndim == 2
                and grid.ndim == 2 and tuple(grid.shape) == (1, 3)
                and raw_pixels.shape[0] > 0 and raw_pixels.shape[1] > 0
                and raw_pixels.is_floating_point()
                and not raw_pixels.requires_grad
                and not grid.requires_grad
                and bool(torch.isfinite(raw_pixels).all().item())
                and bool((grid > 0).all().item()))
        if feature_width is None:
            feature_width = raw_pixels.shape[1]
        require(raw_pixels.shape[1] == feature_width)
        pixels.append(raw_pixels)
        grids.append(grid)
    batch = {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in plan.items()
        if name in ("input_ids", "labels", "attention_mask")
    }
    batch["pixel_values"] = torch.cat(pixels, dim=0)
    batch["image_grid_thw"] = torch.cat(grids, dim=0)
    require(tuple(batch["input_ids"].shape) ==
            tuple(batch["labels"].shape) ==
            tuple(batch["attention_mask"].shape) ==
            (len(examples), plan["max_length"])
            and tuple(batch["image_grid_thw"].shape) == (len(examples), 3)
            and batch["pixel_values"].shape[0] ==
            sum(p.shape[0] for p in pixels)
            and int(batch["attention_mask"].sum().item()) ==
            sum(plan["sequence_lengths"])
            and int((batch["labels"] != -100).sum().item()) ==
            sum(plan["supervised_token_counts"])
            and not bool(((batch["attention_mask"] == 0) &
                          (batch["labels"] != -100)).any().item()))
    # No gradients, model, loss computation, checkpoint, or optimizer call.
    return batch

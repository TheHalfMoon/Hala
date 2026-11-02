"""H369 source-only proof of pinned Qwen3-VL image-token cardinality.

The exact eight cached public Qwen processor/tokenizer files are pinned
and SHA-verified in the owner REAL CPU runner. This module does not load
models, images, or torch. It grants no legal TRAIN or blind-OOD rights.
"""
from __future__ import annotations

# Verified on actual owner-local PINNED Qwen3VLProcessor revision
# 89644892e4d85e24eaac8bacfd4f463576704203, synthetic solid RGB
# 960x720: grid=[1,44,60], 2640 patches, 660 image token IDs.
PINNED_IMAGE_TOKEN_ID = 151655
PINNED_SPATIAL_MERGE_SIZE = 2


def image_token_cardinality(
    prefix_ids: list[int], full_ids: list[int], patch_rows: int
) -> int:
    """Demand exactly one image's expected token placeholders in the prompt.

    For Qwen3-VL's pinned processor, each 2x2 group of image patches maps
    to one <|image_pad|> text token. The H364 collator accepts one image
    per source row. This check rejects a forged image-grid/token mismatch
    even if prefix/full strings remain mutually consistent.
    """
    if not (type(prefix_ids) is list and type(full_ids) is list and
            type(patch_rows) is int and patch_rows >= 4 and
            patch_rows % (PINNED_SPATIAL_MERGE_SIZE ** 2) == 0 and
            all(type(t) is int and t >= 0 for t in prefix_ids + full_ids) and
            0 < len(prefix_ids) < len(full_ids) and
            full_ids[:len(prefix_ids)] == prefix_ids):
        raise ValueError("H369_PINNED_IMAGE_TOKEN_CARDINALITY_REJECTED")
    expected = patch_rows // (PINNED_SPATIAL_MERGE_SIZE ** 2)
    if not (prefix_ids.count(PINNED_IMAGE_TOKEN_ID) == expected and
            full_ids.count(PINNED_IMAGE_TOKEN_ID) == expected and
            PINNED_IMAGE_TOKEN_ID not in full_ids[len(prefix_ids):]):
        raise ValueError("H369_PINNED_IMAGE_TOKEN_CARDINALITY_REJECTED")
    return expected

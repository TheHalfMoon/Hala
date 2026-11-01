"""H331: OPTIONAL, immutable first-click visual-instruction variant for Hala.

This is a HYPOTHESIS, not trained-model gain, measured improvement, or blind OOD.
The fixed text uses only visible screenshot conventions and the user task.
It never supplies source DOM, reference points, gold, target positions or labels.
"""
from __future__ import annotations

from h323.dynamic_grammar import (
    STAGE_CLICK, STAGE_TYPE, model_visible_prompt,
)

CONTROL = "control"
VISUAL_RTL = "visual_target_rtl"
VARIANTS = frozenset((CONTROL, VISUAL_RTL))

# Keep the fixed experiment change strictly at the first visual click.
# The Arabic/right-to-left coordinate distinction is geometry, not
# inversion: image x=0 is left and x=999 right for either language.
FIRST_CLICK_ONLY = (
    " Compare the station identifier written in the user task against "
    "the text actually printed on each visible STATION BUTTON in the "
    "screenshot. Match the entire identifier, not visual reading order "
    "or card position. In an RTL layout, text direction changes but "
    "image x coordinates do NOT flip: x=0 is the screenshot's LEFT "
    "edge and x=999 its RIGHT edge. For the matched visible button, "
    "predict the center of its clickable rectangle from image pixels. "
    "Do not click the nearby label, container or a different station. "
    "If the screenshot does not let you identify the matching button "
    "confidently, abstain. This is an image-only task: no DOM, source "
    "metadata, hidden reference points or oracle results."
)


def build_stage_prompt(task: str, stage: str, *, variant: str) -> str:
    """Only the explicit candidate changes first-click prompt instructions.

    H323 control remains byte-for-byte identical. Stage-two prompt is
    *always* identical across variants to isolate the first-click hypothesis.
    """
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("H331 UNKNOWN_PROMPT_VARIANT_REFUSED")
    baseline = model_visible_prompt(task, stage)
    if variant == CONTROL or stage == STAGE_TYPE:
        return baseline
    if stage != STAGE_CLICK:
        raise ValueError("H331 INVALID_STAGE")
    candidate = baseline + FIRST_CLICK_ONLY
    if len(candidate) > 2000:
        raise ValueError("H331 PROMPT_TOO_LONG")
    return candidate

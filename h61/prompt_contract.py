"""H61 the one native drag prompt contract used by training and by evaluation.

The H60 baseline was measured with the public DEV runner prompt and the H60
curriculum was built with a different, similar-looking prompt. Nothing proved
the two were the same contract, so a measured training gain could have been a
prompt mismatch rather than a grounding gain.

This module owns the single template. `drag_prompt` reproduces the H60 public
DEV prompt byte for byte for the same body, image size and ordered flag, and
`tests/test_h61_prompt_contract.py` asserts that equivalence against the H60
runner itself. Training rows and the public DEV evaluation therefore share one
contract by construction.
"""
from __future__ import annotations

PROMPT_REVISION = "hala-h61-native-drag-prompt-contract-v1"
CONTRACT_SUFFIX = (
    "Return ONLY a JSON object with type=drag, start_point=[x,y], end_point=[x,y], "
    "coordinate_space=source_image_pixels, ordered={ordered}. "
    "Coordinates must be integer pixels of the ORIGINAL screenshot; "
    "do not click, navigate, or execute anything."
)
MAX_BODY = 1200


def drag_prompt(body: str, width: int, height: int, ordered: bool) -> str:
    """One native drag prompt for one original screenshot.

    `body` is the natural-language task, `width`/`height` are the ORIGINAL
    screenshot dimensions and `ordered` is the original source flag.
    """
    if not isinstance(body, str) or not 4 <= len(body) <= MAX_BODY:
        raise ValueError("Bounded natural-language task body required")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Original screenshot dimensions required")
    if type(ordered) is not bool:
        raise ValueError("Original ordered flag must be boolean")
    return (
        f"{body}\n"
        f"The image is {width}x{height} pixels. "
        + CONTRACT_SUFFIX.format(ordered="true" if ordered else "false")
    )


def source_body(intent: str, ordered: bool) -> str:
    """The original source instruction as the task body of the contract.

    The original DragOn instruction already states the whole task, so it is
    used verbatim. Nothing is translated, summarised or rewritten.
    """
    if not isinstance(intent, str) or not 4 <= len(intent) <= MAX_BODY:
        raise ValueError("Bounded original instruction required")
    if type(ordered) is not bool:
        raise ValueError("Original ordered flag must be boolean")
    direction = "ordered" if ordered else "unordered"
    return f"{intent}\nReturn only ONE {direction} drag action in original-image pixel coordinates."

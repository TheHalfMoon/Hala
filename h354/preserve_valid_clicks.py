"""H354 reduce interventions: preserve a click already within any visual button.

H319 screenshot-only, model-free original synthetic source research.
No DOM, gold, task text, target IDs, source stylesheet or OCR enters
this rule. Exact frozen H352 RGB detector is reused, no threshold search.
"""
from __future__ import annotations

from h199.native_action_sft import parse_actions
from h352.pixel_button_snap import (
    pixel_button_rectangles, snap_native_first_click as near_center_snap,
)


def preserve_valid_pixel_click(png: bytes, native: str) -> str:
    if type(native) is not str or len(native) > 256:
        raise ValueError("H354 NATIVE_ACTION_REFUSED")
    actions = parse_actions(native)["actions"]
    if len(actions) != 1 or actions[0]["kind"] not in ("click", "abstain"):
        raise ValueError("H354 NATIVE_ACTION_REFUSED")
    if actions[0]["kind"] == "abstain":
        return native
    x, y = actions[0]["point"]
    boxes = pixel_button_rectangles(png)
    # A click inside ANY detected button (even the wrong button) is not
    # modified; choosing which button is correct requires a task/oracle.
    if any(left <= x < right and top <= y < bottom
           for left, top, right, bottom in boxes):
        return native
    # Only an unambiguous near-but-outside click is moved to center.
    return near_center_snap(png, native)


FROZEN_POLICY_SUFFIX = '''

# H354 predeclared conservative policy; the screenshot detector above is
# unchanged frozen H352 source. No gold/DOM/task labels enter this wrapper.
_old_unconditional_near_snap = snap_native_first_click

def snap_native_first_click(png: bytes, native: str) -> str:
    need(type(native) is str and len(native) <= 256)
    actions = parse_actions(native)["actions"]
    need(len(actions) == 1 and actions[0]["kind"] in {"click", "abstain"})
    if actions[0]["kind"] == "abstain":
        return native
    x, y = actions[0]["point"]
    boxes = pixel_button_rectangles(png)
    if any(left <= x < right and top <= y < bottom
           for left, top, right, bottom in boxes):
        return native
    return _old_unconditional_near_snap(png, native)
'''

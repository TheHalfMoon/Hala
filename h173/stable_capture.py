"""Fail-closed stabilization of original-browser pre-action screenshots.

This module never changes original upstream source, pauses CSS, injects
styles, or passes reference coordinates to the model. It only normalizes
the *external pointer state* and checks two decoded pixel rasters.
A successful guard is NOT a qualified OOD result.
"""
from io import BytesIO
import hashlib

from PIL import Image

VIEWPORT = (960, 720)
NEUTRAL_POINT = (0, 0)


def original_unmodified_png(page) -> bytes:
    """Capture the original app with native CSS animations allowed.

    Unlike the historical H159 helper, this MUST NOT pass
    ``animations='disabled'``: Playwright fast-forwards finite animations
    and cancels infinite ones in that mode, changing the rendered state.
    """
    return page.screenshot(type="png", animations="allow")


def same_original_pixels(first: bytes, second: bytes) -> bool:
    """PNG encoding/compression may differ; RGBA pixels may not."""
    try:
        with Image.open(BytesIO(first)) as a, Image.open(BytesIO(second)) as b:
            if a.format != "PNG" or b.format != "PNG":
                return False
            if a.size != VIEWPORT or b.size != VIEWPORT:
                return False
            a.load()
            b.load()
            return a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes()
    except (OSError, ValueError, TypeError):
        return False


def original_visual_hash(raw: bytes) -> str:
    """SHA-256 of visible RGBA pixel data (independent of PNG compression)."""
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG" or image.size != VIEWPORT:
            raise ValueError("H173 full size original PNG required")
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def stabilized_original_png(page, screenshot, *, modal=False) -> bytes:
    """Normalize pointer *after* scrolling the target, then fail closed.

    screenshot is a callable e.g. h159.stable_source_qa.stable_png.
    No cached image, no selection of a better-looking frame: exactly two
    consecutive frames must agree and hovered modal controls are forbidden.
    """
    page.mouse.move(*NEUTRAL_POINT)
    # Allow original finite open/slide transitions to settle naturally.
    # No CSS injection, animation cancellation, or best-frame selection.
    page.wait_for_timeout(650)
    if modal and page.locator(
        ".tingle-modal--visible .tingle-modal__close:hover"
    ).count() != 0:
        raise ValueError("H173 Tingle close button still hovered")
    first = screenshot(page)
    page.wait_for_timeout(160)
    second = screenshot(page)
    if not same_original_pixels(first, second):
        raise ValueError(
            "H173 original app not pixel-stable after pointer normalization"
        )
    return second

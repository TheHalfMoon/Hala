"""H173 deterministic original screenshot state guard; no model calls."""
import io

import pytest
from PIL import Image

from h173.stable_capture import (
    NEUTRAL_POINT,
    same_original_pixels,
    stabilized_original_png,
)


def png(rgb, *, compression=6, size=(960, 720)):
    img = Image.new("RGBA", size, rgb)
    out = io.BytesIO()
    img.save(out, format="PNG", compress_level=compression)
    return out.getvalue()


class Mouse:
    def __init__(self, page):
        self.page = page
        self.moves = []

    def move(self, x, y):
        self.moves.append((x, y))
        self.page.hover = False


class ModalLocator:
    def __init__(self, page):
        self.page = page

    def count(self):
        return int(self.page.hover)


class Page:
    def __init__(self):
        self.hover = True
        self.mouse = Mouse(self)
        self.timeouts = []
        self.selectors = []

    def wait_for_timeout(self, ms):
        self.timeouts.append(ms)

    def locator(self, selector):
        self.selectors.append(selector)
        return ModalLocator(self)


def test_identical_rgba_with_different_png_encoding_is_safe():
    a = png((5, 20, 200, 255), compression=0)
    b = png((5, 20, 200, 255), compression=9)
    assert a != b
    assert same_original_pixels(a, b)


@pytest.mark.parametrize("other", [
    png((5, 20, 201, 255)),
    png((5, 20, 200, 254)),
    png((5, 20, 200, 255), size=(700, 960)),
    b"not a png",
])
def test_any_real_original_visual_change_fails_closed(other):
    assert not same_original_pixels(png((5, 20, 200, 255)), other)


def test_necessary_pointer_normalization_precedes_original_snapshot():
    page = Page()
    frames = []

    def render(p):
        frames.append(p.hover)
        return png((255, 0, 0, 255) if p.hover else (0, 120, 10, 255))

    result = stabilized_original_png(page, render, modal=True)
    assert result == png((0, 120, 10, 255))
    assert frames == [False, False]
    assert page.mouse.moves == [NEUTRAL_POINT]
    assert len(page.timeouts) == 2
    assert page.selectors == [
        ".tingle-modal--visible .tingle-modal__close:hover"
    ]


def test_two_different_consecutive_frames_hard_fail():
    page = Page()
    before = png((0, 0, 0, 255))
    after = png((255, 255, 255, 255))
    count = [0]

    def unstable(_):
        count[0] += 1
        return before if count[0] == 1 else after

    with pytest.raises(ValueError, match="pixel-stable"):
        stabilized_original_png(page, unstable, modal=True)


def test_still_hovered_modal_control_hard_fails():
    page = Page()
    page.mouse.move = lambda *_: None  # simulate a broken host controller
    with pytest.raises(ValueError, match="hovered"):
        stabilized_original_png(
            page, lambda p: png((0, 0, 0, 255)), modal=True
        )

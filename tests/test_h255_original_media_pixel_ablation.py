"""Synthetic-only H255 tests: no real media, network or owner data."""
import io

import pytest
from PIL import Image

from h255.original_media_pixel_ablation import changed_pixels, classify


BOX = {"x": 10, "y": 10, "width": 20, "height": 20}


def png(changes=()):
    image = Image.new("RGB", (960, 720), (255, 255, 255))
    for x, y in changes:
        image.putpixel((x, y), (0, 0, 0))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def test_exact_inside_outside_pixel_counts():
    assert changed_pixels(png(), png([(15, 15), (17, 17), (50, 50)]), BOX) == {
        "inside_changed_pixels": 2, "outside_changed_pixels": 1}


def test_control_stability_required():
    z = {"inside_changed_pixels": 0, "outside_changed_pixels": 0}
    one = {"inside_changed_pixels": 1, "outside_changed_pixels": 0}
    assert classify(one, one, z) == "INDETERMINATE_DYNAMIC_PAGE"
    assert classify(z, one, one) == "INDETERMINATE_DYNAMIC_PAGE"


def test_visible_dependency_requires_isolated_change():
    z = {"inside_changed_pixels": 0, "outside_changed_pixels": 0}
    inside = {"inside_changed_pixels": 5, "outside_changed_pixels": 0}
    outside = {"inside_changed_pixels": 5, "outside_changed_pixels": 1}
    assert classify(z, inside, z) == "VISIBLE_PIXEL_DEPENDENCY_OBSERVED"
    assert classify(z, outside, z) == "VISIBLE_DEPENDENCY_OUTSIDE_IMAGE_RECT"
    assert classify(z, z, z) == "NO_VISIBLE_PIXEL_EFFECT"


@pytest.mark.parametrize("box", [
    {"x": -1, "y": 0, "width": 5, "height": 5},
    {"x": 950, "y": 0, "width": 20, "height": 5},
    {"x": 0, "y": 0, "width": float("nan"), "height": 5},
    {"x": 0, "y": 0, "width": True, "height": 5},
])
def test_bad_boxes_fail_closed(box):
    with pytest.raises(ValueError):
        changed_pixels(png(), png(), box)


def test_invalid_counts_fail_closed():
    z = {"inside_changed_pixels": 0, "outside_changed_pixels": 0}
    with pytest.raises(ValueError):
        classify(z, {"inside_changed_pixels": -1, "outside_changed_pixels": 0}, z)

# CI's existing unittest discovery runs these synthetic checks without a new workflow.
import unittest


class H255HostedUnittest(unittest.TestCase):
    def test_pixel_diff_isolated_and_outside(self):
        self.assertEqual(
            changed_pixels(png(), png([(15, 15), (50, 50)]), BOX),
            {"inside_changed_pixels": 1, "outside_changed_pixels": 1},
        )

    def test_visible_outside_effect_requires_stable_controls(self):
        z = {"inside_changed_pixels": 0, "outside_changed_pixels": 0}
        effect = {"inside_changed_pixels": 1, "outside_changed_pixels": 1}
        self.assertEqual(classify(z, effect, z), "VISIBLE_DEPENDENCY_OUTSIDE_IMAGE_RECT")
        self.assertEqual(classify(effect, effect, z), "INDETERMINATE_DYNAMIC_PAGE")

    def test_never_admit_without_a_real_difference(self):
        z = {"inside_changed_pixels": 0, "outside_changed_pixels": 0}
        self.assertEqual(classify(z, z, z), "NO_VISIBLE_PIXEL_EFFECT")

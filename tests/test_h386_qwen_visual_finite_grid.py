"""H386: finite image tensor and positive Qwen vision grid are mandatory.

Synthetic fake tensor contract only; model/weights/GPU/OOD/training forbidden.
"""
from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch

from h216.real_qwen_processor_alignment import visual_tensor_receipt


class Tensor:
    def __init__(self, values):
        self.values = values
    def numel(self):
        return len(self.values)
    def tolist(self):
        return self.values


class Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class Mask:
    def __init__(self, values):
        self.values = values
    def all(self):
        return Scalar(all(self.values))


def fake_torch():
    return types.SimpleNamespace(
        equal=lambda a, b: a.values == b.values,
        isfinite=lambda tensor: Mask([
            math.isfinite(v) for v in tensor.values
        ]),
    )


def batches(pixels, grid):
    return (
        {"pixel_values": Tensor(list(pixels)),
         "image_grid_thw": Tensor(list(grid))},
        {"pixel_values": Tensor(list(pixels)),
         "image_grid_thw": Tensor(list(grid))},
    )


class H386VisualFiniteGridTests(unittest.TestCase):
    def setUp(self):
        self.mock = patch.dict(sys.modules, {"torch": fake_torch()})
        self.mock.start()
    def tearDown(self):
        self.mock.stop()

    def test_equal_positive_finite_fake_tensors_pass(self):
        for grid in ([1, 2, 3], [[1, 2, 3]]):
            with self.subTest(grid=grid):
                self.assertEqual(visual_tensor_receipt(
                    *batches([0.2, -0.5, 0.0, 1.5], grid)
                ), 4)

    def test_infinite_pixel_values_rejected_even_when_exactly_equal(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    visual_tensor_receipt(*batches([1.0, value, 0.1], [1,2,3]))

    def test_zero_negative_or_nonintegral_image_grid_rejected(self):
        for grid in ([0, 2, 3], [1, 0, 3], [1, 2, -1],
                     [1, 3.5, 8], [True, 2, 3], [[1, 2, 0]],
                     [1, 2], [[1,2,3],[1,2,3]]):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    visual_tensor_receipt(*batches([1.0,2.0,3.0],grid))

    def test_visual_grid_changed_is_refused_without_model(self):
        before, after = batches([1.0,2.0], [1,2,3])
        after["image_grid_thw"] = Tensor([1,2,4])
        with self.assertRaises(ValueError):
            visual_tensor_receipt(before,after)

    def test_finite_but_changed_pixels_refused(self):
        before, after = batches([1.0,2.0], [1,2,3])
        after["pixel_values"] = Tensor([1.0,3.0])
        with self.assertRaises(ValueError):
            visual_tensor_receipt(before,after)


if __name__ == "__main__":
    unittest.main()

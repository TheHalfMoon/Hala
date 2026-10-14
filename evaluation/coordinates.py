"""Coordinates use captured pixels; DPI scale is metadata, never divided twice."""
import math


def _numbers(values):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite numbers, not booleans")


def normalize_box(box, width, height):
    if len(box) != 4:
        raise ValueError("Expected x,y,width,height")
    _numbers([*box, width, height])
    x, y, w, h = box
    if width <= 0 or height <= 0 or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("Invalid captured-pixel bounds")
    return [x / width, y / height, w / width, h / height]


def denormalize_box(box, width, height):
    if len(box) != 4:
        raise ValueError("Expected normalized x,y,width,height")
    _numbers([*box, width, height])
    x, y, w, h = box
    if width <= 0 or height <= 0 or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
        raise ValueError("Invalid normalized bounds")
    return [x * width, y * height, w * width, h * height]

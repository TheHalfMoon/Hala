"""Independent geometry validation and H29 full-observed-set point binding."""
from __future__ import annotations

import math


def validate_bounds(bounds: list, width: int, height: int) -> None:
    if type(bounds) is not list or len(bounds) != 4:
        raise ValueError("bounds must be [x,y,width,height]")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds):
        raise ValueError("bounds must be finite numbers")
    x, y, w, h = bounds
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("bounds outside frame")


def contains(bounds: list, point: tuple[float, float]) -> bool:
    x, y, w, h = bounds
    px, py = point
    return x <= px < x + w and y <= py < y + h


def bind_point(point: tuple[float, float], observed: list[dict], eligible: list[dict], width: int, height: int):
    for item in observed:
        validate_bounds(item["bounds"], width, height)
    observed_hits = [item for item in observed if contains(item["bounds"], point)]
    eligible_hits = [item for item in eligible if contains(item["bounds"], point)]
    if len(observed_hits) != 1 or len(eligible_hits) != 1 or observed_hits[0] is not eligible_hits[0]:
        return None
    return observed_hits[0]

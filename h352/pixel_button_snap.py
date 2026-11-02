"""H352: screenshot-PX-ONLY first-click button-center hypothesis.

PUBLIC EXPOSED H319 SOURCE RESEARCH ONLY, 960x720. There is no OCR,
target ID, task/gold label, DOM, stylesheet, network, live application,
source oracle or model text inside this candidate. Fail closed when the
visual pattern is ambiguous. This is NOT generally safe computer use and
NOT learned Qwen adaptation or blind OOD.
"""
from __future__ import annotations

from io import BytesIO
from math import hypot
import json

from h199.native_action_sft import parse_actions

WIDTH, HEIGHT = 960, 720
MAX_PNG = 4 * 1024 * 1024
MAX_SNAP_DISTANCE = 48
AMBIGUITY_MARGIN = 18


def need(value: bool) -> None:
    if not value:
        raise ValueError("H352 PIXEL_ONLY_BUTTON_SNAP_REFUSED")


def _banded_runs(pixels, color: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    """Extract four rectangles from anonymous repeated RGB regions only."""
    selected_rows = []
    for y in range(150, 665):
        count = sum(pixels[x, y] == color for x in range(35, 925))
        if count >= 400:
            selected_rows.append(y)
    bands: list[list[int]] = []
    for y in selected_rows:
        if not bands or y > bands[-1][-1] + 1:
            bands.append([y])
        else:
            bands[-1].append(y)
    bands = [band for band in bands if len(band) >= 25]
    need(len(bands) == 2 and bands[0][-1] + 25 < bands[1][0])
    boxes = []
    for band in bands:
        y_sample = band[0] + 6
        starts = []
        start = None
        for x in range(35, 926):
            colored = x < 925 and pixels[x, y_sample] == color
            if colored and start is None:
                start = x
            elif not colored and start is not None:
                if x - start >= 150:
                    starts.append((start, x))
                start = None
        need(len(starts) == 2)
        for left, right in starts:
            need(200 <= right - left <= 470
                 and 35 <= len(band) <= 64)
            boxes.append((left, band[0], right, band[-1] + 1))
    need(len(boxes) == 4)
    return boxes


def pixel_button_rectangles(png: bytes) -> list[tuple[int, int, int, int]]:
    """Detect 4 large repeated saturated colored UI regions from PNG bytes.

    This preregistered H319-only hypothesis is deliberately conservative.
    An unrelated page, alternate aspect ratio, ambiguous foreground color,
    altered button layout or insufficient rectangle evidence MUST refuse.
    """
    from PIL import Image
    need(type(png) is bytes and 100 <= len(png) <= MAX_PNG
         and png.startswith(b"\x89PNG\r\n\x1a\n"))
    try:
        with Image.open(BytesIO(png)) as image:
            need(image.format == "PNG" and image.size == (WIDTH, HEIGHT))
            rgb = image.convert("RGB")
            rgb.load()
        histogram = rgb.getcolors(maxcolors=200000)
        need(histogram is not None)
        candidate_colors = [
            color for count, color in histogram
            if 40000 <= count <= 110000
            and max(color) - min(color) > 45
        ]
        possible = []
        px = rgb.load()
        for color in candidate_colors:
            try:
                possible.append(_banded_runs(px, color))
            except ValueError:
                pass
        need(len(possible) == 1)
        return possible[0]
    except (OSError, AttributeError, RuntimeError, TypeError):
        raise ValueError("H352 INVALID_OR_AMBIGUOUS_SOURCE_SCREENSHOT") from None


def _distance(point: list[int], rect: tuple[int, int, int, int]) -> float:
    x, y = point
    left, top, right, bottom = rect
    return hypot(max(left - x, 0, x - right),
                 max(top - y, 0, y - bottom))


def snap_native_first_click(png: bytes, native: str) -> str:
    """Only model screenshot and its ONE H199 action enter this candidate.

    No DOM/gold/task text. On uncertain source geometry, refuse rather than
    guess or silently change a click. Unambiguous proposals outside the
    48px neighborhood are left unchanged. Abstention is preserved.
    """
    need(type(native) is str and len(native) <= 256)
    actions = parse_actions(native)["actions"]
    need(len(actions) == 1)
    action = actions[0]
    need(action["kind"] in {"click", "abstain"})
    if action["kind"] == "abstain":
        return native
    boxes = pixel_button_rectangles(png)
    distances = sorted(
        ((_distance(action["point"], box), box) for box in boxes),
        key=lambda row: row[0],
    )
    closest, box = distances[0]
    if closest > MAX_SNAP_DISTANCE:
        return native
    # Never guess a target in a crowded/ambiguous neighborhood.
    if distances[1][0] <= closest + AMBIGUITY_MARGIN:
        return native
    left, top, right, bottom = box
    proposed = [
        (left + right) // 2,
        (top + bottom) // 2,
    ]
    # A canonicalized JSON string is not a geometric intervention.
    # Preserve the exact model action when its point already is the center.
    if proposed == action["point"]:
        return native
    revised = json.dumps(
        {"actions": [{"kind": "click", "point": proposed}]},
        separators=(",", ":"))
    parse_actions(revised)
    return revised

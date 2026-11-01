"""H326 independent source-only geometry audit for H324 RTL first-click failure.

Validates the existing H273 normalized model-point -> pixel conversion
against native source browser hit-testing on publicly exposed H319
BeaconBoard scenes, including RTL. This NEVER executes Qwen or uses model
predictions and must not be presented as Hala model accuracy.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions
from h263.native_browser_executor import execute_one
from h273.relative_native_bridge import normalize_strict_action, normalized_to_pixel
from h319.beaconboard_dynamic_research import (
    IDS, ROLE as EXPOSED_ROLE, SCENES, checked_source,
    inspect_all_stations, original_scene,
)
from h322.source_capture import source_capture_png

ROLE = "H326_PUBLIC_EXPOSED_SOURCE_GEOMETRY_AUDIT_ONLY"
MAX_SCENES = 12


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H326 SOURCE_GEOMETRY_AUDIT_REFUSED")


def normalized_center(box: dict, extent: int, axis: str) -> int:
    """Convert center of source-owned rendered button into Qwen's 0..999 grid.

    Geometry is a source TEST ORACLE only, never a model/policy input.
    """
    need(type(box) is dict and set(box) >= {"x", "y", "width", "height"}
         and axis in ("x", "y")
         and extent in (WIDTH, HEIGHT))
    value, size = box[axis], box["width" if axis == "x" else "height"]
    need(type(value) in (int, float) and type(size) in (int, float)
         and size >= 30 and 0 <= value < extent
         and value + size <= extent)
    center = value + size / 2
    need(0 < center < extent)
    grid = min(999, max(0, round(center * 1000 / extent)))
    native = normalized_to_pixel(grid, extent)
    need(value < native < value + size)
    return grid


def source_case(browser: Any, source: Path, scene: tuple,
                *, x_reflection_negative_control: bool = False) -> dict:
    need(type(scene) is tuple and scene in SCENES
         and type(x_reflection_negative_control) is bool)
    _, url = checked_source(source)
    target = IDS[scene[4]]
    with original_scene(browser, source, url, scene) as page:
        empty = {station: (False, "") for station in IDS}
        need(inspect_all_stations(page) == empty)
        visible = page.locator('button[data-station="' + target + '"]')
        need(visible.count() == 1 and visible.is_visible())
        box = visible.bounding_box()
        need(box is not None)
        x = normalized_center(box, WIDTH, "x")
        y = normalized_center(box, HEIGHT, "y")
        if x_reflection_negative_control:
            x = 999 - x
        # Convert only via H273's existing production-normalized adapter.
        original = json.dumps({"actions": [{"kind": "click",
                           "point": [x, y]}]}, separators=(",", ":"))
        native = normalize_strict_action(original)
        parse_actions(native)
        # Original-source screenshot lock is checked BEFORE dispatch.
        first = source_capture_png(page)
        need(type(first) is bytes and first.startswith(b"\x89PNG\r\n\x1a\n")
             and sha256(first).digest() ==
                 sha256(source_capture_png(page)).digest()
             and page.url == url)
        outcome = execute_one(page, native)
        need(outcome == {"kind": "click", "executed": True})
        state = inspect_all_stations(page)
        want = {station: (station == target, "") for station in IDS}
        focused = page.evaluate("""() => {
            const element = document.activeElement;
            return element instanceof HTMLInputElement
                ? element.dataset.station : null;
        }""")
        success = state == want and focused == target
        # Only a boolean/categorical result leaves this trusted source oracle.
        return {
            "language": scene[1], "correct_focus_and_target_state": success,
            "negative_reflection_control": x_reflection_negative_control,
            "source_public_exposed": True,
            "real_model_generations": 0,
            "official_train_dev_ood_admitted": False,
            "independent_blind_ood": False,
            "adapter_gain_proven": False,
        }


def audit_source_geometry(browser: Any, source: Path, *, role: str) -> dict:
    need(type(role) is str and role == ROLE)
    need(isinstance(source, Path))
    checked_source(source)
    rows = [source_case(browser, source, scene) for scene in SCENES]
    need(len(rows) == MAX_SCENES and
         sum(row["language"] == "en" for row in rows) == 6 and
         sum(row["language"] == "ar" for row in rows) == 6 and
         all(row["correct_focus_and_target_state"] for row in rows))
    # A deliberate RTL x-axis inversion must NOT hit the intended button:
    # this is a negative control, not a model response and is never a
    # candidate sample or a scored OOD task.
    reflected = [source_case(browser, source, scene,
                 x_reflection_negative_control=True)
                 for scene in SCENES if scene[1] == "ar"]
    need(len(reflected) == 6 and
         all(not row["correct_focus_and_target_state"] for row in reflected))
    return {
        "status": ROLE,
        "positive_source_cases": len(rows),
        "positive_source_cases_success": sum(
            row["correct_focus_and_target_state"] for row in rows),
        "english_positive_cases": 6,
        "arabic_rtl_positive_cases": 6,
        "rtl_x_reflection_negative_controls": len(reflected),
        "negative_controls_wrong_target": len(reflected),
        "production_h273_normalized_mapping_used": True,
        "genuine_qwen_generations": 0,
        "official_train_dev_ood_admitted": False,
        "independent_blind_ood": False,
        "adapted_gain_proven": False,
        "can_release": False,
    }

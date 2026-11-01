"""H337: source-only distance-band diagnosis for missed first GUI clicks.

H336 genuine owner-private FREE T4 original-source pretrained Qwen
observed EN 6/6 correct focus clicks, AR 4/6 correct focus and 2/6
NO_STATION_OPEN; no wrong-station or focus-loss errors. This module
can test a precise HYPOTHESIS on *future* permitted source-exposed
experiments: were failed predictions near target boundaries or far?

Source DOM/reference point/box/model native click remain trusted
evaluator-only. No raw coordinates, images, target IDs, gold, model
predictions, or per-scene results are exported. A CPU mock callback
is NOT evidence of Qwen skill or improvement.
"""
from __future__ import annotations

from math import hypot, isfinite
from pathlib import Path
from typing import Any, Callable
import json

from h319.beaconboard_dynamic_research import (
    IDS, SCENES, _observed_predict, checked_source, exact_task,
    inspect_all_stations, original_scene,
)
from h263.native_browser_executor import execute_one
from h335.original_first_click_error_attribution import classify_first_click

ROLE = "H337_EXPOSED_SOURCE_ONLY_FIRST_CLICK_MISS_GEOMETRY"
CATEGORIES = (
    "correct_open_and_focus",
    "correct_open_wrong_focus",
    "wrong_station_open",
    "abstain",
    "miss_inside_expected_button",
    "miss_near_target_0_to_12px",
    "miss_offset_target_12_to_48px",
    "miss_farther_than_48px",
)
ORDER = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED")


def _check_rect(rect: object) -> tuple[float, float, float, float]:
    need(type(rect) is dict and set(rect) == {
        "x", "y", "width", "height"})
    for val in rect.values():
        need(type(val) in (int, float) and isfinite(val))
    x, y, w, h = (
        float(rect["x"]), float(rect["y"]),
        float(rect["width"]), float(rect["height"]),
    )
    need(0 <= x < 960 and 0 <= y < 720
         and 0 < w <= 960 - x and 0 < h <= 720 - y)
    return x, y, x + w, y + h


def classify_source_missed_point(
    point: object, target: object, all_buttons: object,
) -> str:
    """Source-evaluator-only geometry. Return EXACTLY one coarse bucket.

    A valid no-station-open click may fall inside the expected visible
    rectangle but not activate the JS listener (overlay/inert state).
    Other-button intersection cannot be a target-specific near miss
    without verifying no station was actually opened, done by caller.
    """
    need(type(point) is list and len(point) == 2
         and all(type(p) is int and 0 <= p <= limit
                 for p, limit in zip(point, (959, 719), strict=True))
         and type(all_buttons) is dict and set(all_buttons) == set(IDS))
    bounds = {station: _check_rect(box)
              for station, box in all_buttons.items()}
    need(type(target) is str and target in bounds)
    px, py = point
    left, top, right, bottom = bounds[target]
    dx = max(left - px, 0.0, px - right)
    dy = max(top - py, 0.0, py - bottom)
    distance = hypot(dx, dy)
    need(isfinite(distance))
    if distance == 0:
        return "miss_inside_expected_button"
    if distance <= 12:
        return "miss_near_target_0_to_12px"
    if distance <= 48:
        return "miss_offset_target_12_to_48px"
    return "miss_farther_than_48px"


def source_first_click_distance(
    browser: Any, source: Path, scene: tuple,
    predict: Callable[[bytes, str], str], *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE
         and isinstance(source, Path)
         and type(scene) is tuple and scene in SCENES
         and callable(predict))
    try:
        source, url = checked_source(source)
        target = IDS[scene[4]]
        with original_scene(browser, source, url, scene) as page:
            need(inspect_all_stations(page)
                 == {id_: (False, "") for id_ in IDS})
            action = _observed_predict(page, predict, exact_task(scene), set())
            need(action["kind"] in ("click", "abstain"))
            if action["kind"] == "abstain":
                need(inspect_all_stations(page)
                     == {id_: (False, "") for id_ in IDS})
                cat = "abstain"
                clicks = 0
            else:
                outcome = execute_one(
                    page, json.dumps({"actions": [action]},
                                     separators=(",", ":")))
                need(outcome == {"kind": "click", "executed": True})
                state = inspect_all_stations(page)
                focus = page.evaluate("""() => {
                    const e = document.activeElement;
                    return e instanceof HTMLInputElement
                        ? e.dataset.station : null;
                }""")
                state_cat = classify_first_click(state, focus, target)
                if state_cat == "no_station_open":
                    boxes = {station: page.locator(
                        'button[data-station="' + station + '"]'
                    ).bounding_box() for station in IDS}
                    cat = classify_source_missed_point(
                        action["point"], target, boxes)
                else:
                    cat = state_cat
                clicks = 1
            need(page.url == url)
    except Exception:
        raise ValueError("H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED") from None
    need(cat in CATEGORIES)
    return {
        "status": ROLE, "language": scene[1], "category": cat,
        "native_actions_dispatched": clicks,
        "source_public_exposed": True,
        "real_model_identity_attested": False,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "learned_adapter_gain": False, "can_release": False,
    }


def aggregate_source_distance(rows: object) -> dict:
    need(type(rows) is list and len(rows) == 12)
    groups = {lang: {"scenes": 0, "native_actions": 0,
                     "categories": {cat: 0 for cat in CATEGORIES}}
              for lang in ("en", "ar")}
    for row in rows:
        need(type(row) is dict and set(row) == {
            "status", "language", "category",
            "native_actions_dispatched", "source_public_exposed",
            "real_model_identity_attested", "official_train_admitted",
            "fresh_blind_ood", "learned_adapter_gain", "can_release"})
        need(row["status"] == ROLE and row["language"] in groups
             and row["category"] in CATEGORIES
             and type(row["native_actions_dispatched"]) is int
             and row["native_actions_dispatched"]
                 == (0 if row["category"] == "abstain" else 1)
             and row["source_public_exposed"] is True
             and all(row[key] is False for key in (
                 "real_model_identity_attested", "official_train_admitted",
                 "fresh_blind_ood", "learned_adapter_gain", "can_release")))
        group = groups[row["language"]]
        group["scenes"] += 1
        group["native_actions"] += row["native_actions_dispatched"]
        group["categories"][row["category"]] += 1
    need(all(groups[lang]["scenes"] == 6
             and sum(groups[lang]["categories"].values()) == 6
             and groups[lang]["native_actions"]
                 == 6 - groups[lang]["categories"]["abstain"]
             for lang in groups))
    return {
        "status": ROLE, "source_public_exposed_scenes": 12,
        "by_language": groups, "source_only_not_model_attested": True,
        "new_model_generations": 0, "new_optimizer_steps": 0,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "learned_gain_proven": False, "can_release": False,
    }


def audit_source_distance(
    browser: Any, source: Path,
    policy_factory: Callable[[tuple], Callable[[bytes, str], str]],
    *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE
         and callable(policy_factory))
    checked_source(source)
    rows = [
        source_first_click_distance(
            browser, source, SCENES[i], policy_factory(SCENES[i]),
            role=ROLE)
        for i in ORDER
    ]
    return aggregate_source_distance(rows)

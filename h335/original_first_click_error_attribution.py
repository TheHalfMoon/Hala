"""H335: first-click failure attribution from a trusted original GUI source.

The genuine pretrained H333 owner-private exposed research showed Arabic
0/2 in each prompt arm with "wrong_focus_or_state"; H335 provides a
separate **source-only forensic instrument** to disambiguate that broad
category in *future* authorized research. It does not reopen H333 private
predictions, run Qwen, update weights or claim improved accuracy.

The callback sees ONLY a real screenshot and existing task prompt. Source
DOM, target IDs, focus and station state remain evaluator-only. No target
coordinates, actions, images or user/model texts enter returned results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from h199.native_action_sft import source_task_prompt
from h263.native_browser_executor import execute_one
from h319.beaconboard_dynamic_research import (
    IDS, ROLE as ORIGINAL_ROLE, SCENES, _observed_predict,
    checked_source, exact_task, inspect_all_stations, original_scene,
)

ROLE = "H335_PUBLIC_EXPOSED_SOURCE_FIRST_CLICK_DIAGNOSIS_ONLY"
CATEGORIES = (
    "correct_open_and_focus",
    "correct_open_wrong_focus",
    "wrong_station_open",
    "no_station_open",
    "abstain",
)
_LANG = ("en", "ar")


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H335 FIRST_CLICK_SOURCE_DIAGNOSIS_REFUSED")


def classify_first_click(state: object, focus: object, target: object) -> str:
    """Trusted evaluator only. No DOM, target or focus passed to policy."""
    need(type(state) is dict and set(state) == set(IDS)
         and type(target) is str and target in IDS
         and (focus is None or (type(focus) is str and focus in IDS)))
    opened = []
    for station in IDS:
        value = state[station]
        need(type(value) is tuple and len(value) == 2
             and type(value[0]) is bool and type(value[1]) is str
             and value[1] == "")
        if value[0]:
            opened.append(station)
    need(len(opened) <= 1)
    if not opened:
        need(focus is None)
        return "no_station_open"
    if opened[0] != target:
        need(focus in (None, opened[0]))
        return "wrong_station_open"
    # With exactly one open station, a different station's hidden input
    # cannot legitimately own focus in this original offline source.
    need(focus in (None, target))
    return ("correct_open_and_focus" if focus == target
            else "correct_open_wrong_focus")


def source_first_click(
    browser: Any, source_path: Path, scene: tuple,
    predict: Callable[[bytes, str], str], *, role: str,
) -> dict:
    """One native click and original source-state classification.

    Per-case return is a *category only*. Consumers must aggregate and
    must never treat synthetic test callbacks as real VLM predictions.
    """
    need(type(role) is str and role == ROLE
         and type(scene) is tuple and scene in SCENES
         and callable(predict))
    try:
        source, url = checked_source(source_path)
        task = exact_task(scene)
        target = IDS[scene[4]]
        with original_scene(browser, source, url, scene) as page:
            initial = {station: (False, "") for station in IDS}
            need(inspect_all_stations(page) == initial)
            action = _observed_predict(
                page, predict, task, set())
            need(action["kind"] in ("click", "abstain"))
            if action["kind"] == "abstain":
                need(inspect_all_stations(page) == initial)
                category = "abstain"
                dispatched = 0
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
                category = classify_first_click(state, focus, target)
                dispatched = 1
            need(page.url == url)
    except Exception:
        raise ValueError("H335 FIRST_CLICK_SOURCE_DIAGNOSIS_REFUSED") from None
    return {
        "status": ROLE,
        "language": scene[1],
        "category": category,
        "source_public_exposed": True,
        "native_actions_dispatched": dispatched,
        "genuine_model_identity_attested": False,
        "official_train_admitted": False,
        "fresh_blind_ood": False,
        "adapted_model_gain_proven": False,
        "can_release": False,
    }


def aggregate_first_click_categories(rows: object) -> dict:
    """Only bounded two-language totals leave a prospective full-cohort job."""
    need(type(rows) is list and len(rows) == 12)
    by_language = {
        lang: {"scenes": 0, "native_actions": 0,
               "categories": {key: 0 for key in CATEGORIES}}
        for lang in _LANG
    }
    for row in rows:
        need(type(row) is dict and set(row) == {
            "status", "language", "category", "source_public_exposed",
            "native_actions_dispatched", "genuine_model_identity_attested",
            "official_train_admitted", "fresh_blind_ood",
            "adapted_model_gain_proven", "can_release"})
        need(row["status"] == ROLE and row["language"] in _LANG
             and row["category"] in CATEGORIES
             and row["source_public_exposed"] is True
             and all(row[key] is False for key in (
                 "genuine_model_identity_attested", "official_train_admitted",
                 "fresh_blind_ood", "adapted_model_gain_proven",
                 "can_release"))
             and type(row["native_actions_dispatched"]) is int
             and row["native_actions_dispatched"] ==
                (0 if row["category"] == "abstain" else 1))
        group = by_language[row["language"]]
        group["scenes"] += 1
        group["native_actions"] += row["native_actions_dispatched"]
        group["categories"][row["category"]] += 1
    need(all(by_language[lang]["scenes"] == 6 for lang in _LANG)
         and all(sum(by_language[lang]["categories"].values()) == 6
                 for lang in _LANG))
    return {
        "status": ROLE,
        "original_source_scenes": 12,
        "by_language": by_language,
        "source_public_exposed_not_blind": True,
        "real_qwen_model_generations_attested": False,
        "official_train_admitted": False,
        "fresh_blind_ood": False,
        "adapted_gain_proven": False,
        "can_release": False,
    }


def audit_first_click_source(
    browser: Any, source_path: Path,
    predictor_factory: Callable[[tuple], Callable[[bytes, str], str]],
    *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE
         and callable(predictor_factory))
    checked_source(source_path)
    rows = [
        source_first_click(
            browser, source_path, scene, predictor_factory(scene),
            role=ROLE,
        )
        for scene in SCENES
    ]
    return aggregate_first_click_categories(rows)

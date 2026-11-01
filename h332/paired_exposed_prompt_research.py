"""H332: matched, SOURCE-EXPOSED Qwen first-click prompt study protocol.

NO model is loaded here. The only permitted inference caller must attest
its genuine pinned pretrained Qwen, privacy and free GPU separately. Public
BeaconBoard research is *exposed*, NEVER fresh blind OOD, LoRA gain or TRAIN.
All PNG, task text, first-click coordinates and model output remain volatile.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
import json

from h319.beaconboard_dynamic_research import (
    IDS, ROLE as SOURCE_ROLE, SCENES, _model_visible_rgb_digest,
    _observed_png, checked_source, inspect_all_stations, original_scene,
)
from h323.preloaded_dynamic_bridge import run_preloaded_dynamic_source_research
from h331.click_prompt_candidate import CONTROL, VISUAL_RTL, VARIANTS

ROLE = "H332_PRETRAINED_EXPOSED_MATCHED_PROMPT_DEBUG_ONLY"
SCENE_INDICES = (0, 2, 6, 8)
LANGUAGES = ("en", "ar")
OUTCOMES = ("none", "abstain", "wrong_focus_or_state",
            "wrong_code", "source_state_mismatch")
# Counterbalance which prompt is executed first to reduce a systematic
# browser/session cache effect. The same source scene is run independently
# in each arm (never continuing control's state in the candidate arm).
ORDERS = ((CONTROL, VISUAL_RTL), (VISUAL_RTL, CONTROL),
          (VISUAL_RTL, CONTROL), (CONTROL, VISUAL_RTL))


def need(value: bool) -> None:
    if not value:
        raise ValueError("H332 MATCHED_PUBLIC_SOURCE_STUDY_REFUSED")


def valid_observed(data: object) -> tuple[bool, int, int, str]:
    """Fail closed on H323 wrapper protocol, including no false attestations."""
    need(type(data) is dict and set(data) == {
        "status", "source_original_task_success",
        "real_model_generations_claimed_by_wrapper",
        "native_actions_dispatched", "source_categorical_failure",
        "pretrained_model_identity_independently_attested",
        "source_public_exposed", "official_train_dev_ood_admitted",
        "blind_ood_proven", "trained_model_gain_proven", "can_release",
    })
    need(data["status"] ==
         "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY"
         and data["source_public_exposed"] is True
         and data["pretrained_model_identity_independently_attested"] is False
         and data["official_train_dev_ood_admitted"] is False
         and data["blind_ood_proven"] is False
         and data["trained_model_gain_proven"] is False
         and data["can_release"] is False)
    success = data["source_original_task_success"]
    n, dispatched, outcome = (data["real_model_generations_claimed_by_wrapper"],
                              data["native_actions_dispatched"],
                              data["source_categorical_failure"])
    need(type(success) is bool
         and type(n) is int and n in (1, 2)
         and type(dispatched) is int and dispatched in (0, 1, 2)
         and type(outcome) is str and outcome in OUTCOMES
         and success is (outcome == "none"))
    need((n == 1 and dispatched == 0 and outcome == "abstain")
         or (n == 1 and dispatched == 1 and outcome == "wrong_focus_or_state")
         or (n == 2 and dispatched == 1 and outcome in ("abstain", "wrong_code"))
         or (n == 2 and dispatched == 2
             and outcome in ("none", "source_state_mismatch")))
    return success, n, dispatched, outcome


def _first_frame_rgb(browser: Any, source: Path, url: str, scene: tuple) -> bytes:
    """Source-only auditor. The model never receives source geometry/state."""
    need(type(scene) is tuple and scene in SCENES)
    with original_scene(browser, source, url, scene) as page:
        need(inspect_all_stations(page) == {x: (False, "") for x in IDS})
        image = _observed_png(page)
        digest = _model_visible_rgb_digest(image)
        need(digest == _model_visible_rgb_digest(_observed_png(page)))
        return digest


def check_pair_frame_reproducibility(
    browser: Any, source: Path, scene: tuple,
) -> None:
    source, url = checked_source(source)
    first = _first_frame_rgb(browser, source, url, scene)
    second = _first_frame_rgb(browser, source, url, scene)
    need(first == second)


def summarize_pairs(paired: object) -> dict:
    """Process eight strict results and return ONLY bounded aggregates.

    Input is volatile, not serialized. No per-case predictions, exact target,
    prompt text, image digest, button coordinate, or raw gold leaves process.
    """
    need(type(paired) is list and len(paired) == len(SCENE_INDICES))
    counts: dict[tuple[str, str], dict] = {}
    wins = {language: {"control": 0, "candidate": 0, "tie": 0}
            for language in LANGUAGES}
    for i, pair in enumerate(paired):
        language = SCENES[SCENE_INDICES[i]][1]
        need(type(pair) is dict and set(pair) == set(VARIANTS))
        row = {}
        for variant in (CONTROL, VISUAL_RTL):
            success, gen, dispatched, outcome = valid_observed(pair[variant])
            row[variant] = success
            key = (language, variant)
            metrics = counts.setdefault(key, {
                "tasks": 0, "task_success": 0, "model_generations": 0,
                "native_actions": 0,
                "failure_counts": {name: 0 for name in OUTCOMES},
            })
            metrics["tasks"] += 1
            metrics["task_success"] += int(success)
            metrics["model_generations"] += gen
            metrics["native_actions"] += dispatched
            metrics["failure_counts"][outcome] += 1
        if row[CONTROL] == row[VISUAL_RTL]:
            wins[language]["tie"] += 1
        elif row[CONTROL]:
            wins[language]["control"] += 1
        else:
            wins[language]["candidate"] += 1
    aggregate = {}
    for language in LANGUAGES:
        control = counts[(language, CONTROL)]
        candidate = counts[(language, VISUAL_RTL)]
        need(control["tasks"] == candidate["tasks"] == 2
             and sum(wins[language].values()) == 2
             and candidate["task_success"] - control["task_success"] ==
                 wins[language]["candidate"] - wins[language]["control"])
        aggregate[language] = {
            "control": control,
            "candidate": candidate,
            "pair_outcomes": wins[language],
        }
    return {
        "status": ROLE,
        "frozen_public_scene_count": 4,
        "matched_model_source_runs": 8,
        "english_scene_pairs": 2,
        "arabic_rtl_scene_pairs": 2,
        "counterbalanced_orders": True,
        "by_language": aggregate,
        "source_public_exposed_not_blind": True,
        "genuine_model_identity_requires_external_attestation": True,
        "official_train_dev_ood_admitted": False,
        "optimizer_steps": 0,
        "adapted_model_gain_proven": False,
        "fresh_blind_ood_proven": False,
        "can_release": False,
    }


def run_paired_exposed_study(
    browser: Any, source: Path, processor: Any, model: Any,
    *, role: str, runner: Callable[..., dict] = run_preloaded_dynamic_source_research,
) -> dict:
    """Genuine runner ONLY if caller independently verifies loaded Qwen.

    The injected runner exists strictly for invented CPU tests; runner-based
    results are NOT model evidence without separate external attestation.
    """
    need(type(role) is str and role == ROLE and callable(runner))
    checked_source(source)
    paired = []
    for index, order in zip(SCENE_INDICES, ORDERS, strict=True):
        scene = SCENES[index]
        check_pair_frame_reproducibility(browser, source, scene)
        outcomes = {}
        for variant in order:
            if variant == CONTROL:
                # Preserve original H323 constructor/model prompt exactly.
                raw = runner(browser, source, processor, model, scene)
            else:
                raw = runner(browser, source, processor, model, scene,
                             prompt_variant=VISUAL_RTL)
            valid_observed(raw)
            outcomes[variant] = raw
        paired.append(outcomes)
    return summarize_pairs(paired)

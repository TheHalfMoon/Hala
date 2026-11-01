"""H334: frozen paired REAL preloaded visual-policy comparison on EXPOSED GUI.

This callable performs model-time Qwen generations ONLY if an independently
authorized owner-private caller supplies the pinned preloaded processor/model.
It never loads weights, starts a GPU job, writes media, or returns per-case
predictions, task texts, points, images, or gold.
"""
from __future__ import annotations

from h319.beaconboard_dynamic_research import SCENES, BLOB
from h323.preloaded_dynamic_bridge import (
    run_preloaded_dynamic_source_research as _run_arm,
)
from h331.click_prompt_candidate import CONTROL, VISUAL_RTL
from h332.paired_exposed_prompt_research import (
    check_pair_frame_reproducibility as _check_pair,
)

# Freeze all 12 original publicly exposed scenes, alternating languages.
# For each language, each arm runs first on exactly three of six scenes.
SCENE_ORDER = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)
ARM_ORDER = tuple(
    (CONTROL, VISUAL_RTL) if index % 2 == 0
    else (VISUAL_RTL, CONTROL)
    for index in SCENE_ORDER
)
FAILURES = frozenset({
    "none", "abstain", "wrong_focus_or_state",
    "wrong_code", "source_state_mismatch",
})
SAFE_KEYS = frozenset({
    "status", "source_original_task_success",
    "real_model_generations_claimed_by_wrapper",
    "native_actions_dispatched", "source_categorical_failure",
    "pretrained_model_identity_independently_attested",
    "source_public_exposed", "official_train_dev_ood_admitted",
    "blind_ood_proven", "trained_model_gain_proven", "can_release",
})
STATUS = "H334_PAIRED_PRELOADED_PUBLIC_EXPOSED_PROMPT_RESEARCH_ONLY"


class PairedProbeRefused(ValueError):
    """Fixed categorical error, never includes a model or source value."""


def need(condition: bool) -> None:
    if not condition:
        raise PairedProbeRefused("H334 PAIRED_EXPOSED_PROBE_REFUSED")


def _safe_arm(value: object) -> tuple[bool, str, int, int]:
    """Refuse unknown fields and impossible terminal native-action states."""
    need(type(value) is dict and set(value) == SAFE_KEYS)
    for key, expected in (
        ("status", "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY"),
        ("pretrained_model_identity_independently_attested", False),
        ("source_public_exposed", True),
        ("official_train_dev_ood_admitted", False),
        ("blind_ood_proven", False),
        ("trained_model_gain_proven", False),
        ("can_release", False),
    ):
        need(type(value[key]) is type(expected) and value[key] == expected)
    success = value["source_original_task_success"]
    why = value["source_categorical_failure"]
    n = value["real_model_generations_claimed_by_wrapper"]
    actions = value["native_actions_dispatched"]
    need(type(success) is bool and type(why) is str and why in FAILURES
         and type(n) is int and type(actions) is int)
    valid = (
        (n == 1 and actions == 0 and why == "abstain")
        or (n == 1 and actions == 1 and why == "wrong_focus_or_state")
        or (n == 2 and actions == 1 and why in {"abstain", "wrong_code"})
        or (n == 2 and actions == 2
            and why in {"none", "source_state_mismatch"})
    )
    need(valid and success is (why == "none"))
    return success, why, n, actions


def _blank() -> dict[str, int]:
    return {
        "pairs": 0, "control_success": 0, "candidate_success": 0,
        "both_success": 0, "both_failed": 0,
        "candidate_only": 0, "control_only": 0,
        "control_abstain": 0, "candidate_abstain": 0,
        "control_wrong_focus": 0, "candidate_wrong_focus": 0,
        "control_generations": 0, "candidate_generations": 0,
        "control_native_actions": 0, "candidate_native_actions": 0,
        "control_first": 0, "candidate_first": 0,
    }


def run_paired_exposed_preloaded_probe(browser, source, processor, model) -> dict:
    """Run 24 genuine model-policy arms on 12 reset original source scenes.

    No evidence of Qwen identity or free quota is supplied by this function.
    The caller MUST independently preflight the exact source SHA, model
    revision, privacy, noncommercial FREE GPU quota and prospective protocol.
    A failure in any arm aborts without returning partial statistics.
    """
    need(len(SCENES) == 12 and len(SCENE_ORDER) == len(ARM_ORDER) == 12
         and set(SCENE_ORDER) == set(range(12))
         and all(SCENES[i][1] == ("en" if i < 6 else "ar")
                 for i in range(12)))
    totals = _blank()
    by_language = {"en": _blank(), "ar": _blank()}
    try:
        for index, order in zip(SCENE_ORDER, ARM_ORDER, strict=True):
            need(type(order) is tuple and set(order) == {CONTROL, VISUAL_RTL}
                 and len(order) == 2)
            scene = SCENES[index]
            # H332 source-owned exact decoded-RGB replay check before either arm.
            _check_pair(browser, source, scene)
            rows = {}
            for arm in order:
                # H323 creates a fresh isolated offline browser context
                # and a NEW policy instance for each scene/arm.
                raw = _run_arm(
                    browser, source, processor, model, scene,
                    prompt_variant=arm)
                rows[arm] = _safe_arm(raw)
            control, candidate = rows[CONTROL], rows[VISUAL_RTL]
            c, k = control[0], candidate[0]
            delta = {
                "pairs": 1, "control_success": int(c),
                "candidate_success": int(k),
                "both_success": int(c and k),
                "both_failed": int(not c and not k),
                "candidate_only": int(k and not c),
                "control_only": int(c and not k),
                "control_abstain": int(control[1] == "abstain"),
                "candidate_abstain": int(candidate[1] == "abstain"),
                "control_wrong_focus": int(control[1] == "wrong_focus_or_state"),
                "candidate_wrong_focus": int(candidate[1] == "wrong_focus_or_state"),
                "control_generations": control[2],
                "candidate_generations": candidate[2],
                "control_native_actions": control[3],
                "candidate_native_actions": candidate[3],
                "control_first": int(order[0] == CONTROL),
                "candidate_first": int(order[0] == VISUAL_RTL),
            }
            for aggregate in (totals, by_language[scene[1]]):
                for key, value in delta.items():
                    aggregate[key] += value
        need(totals["pairs"] == 12
             and all(by_language[lang]["pairs"] == 6
                     and by_language[lang]["control_first"] == 3
                     and by_language[lang]["candidate_first"] == 3
                     for lang in ("en", "ar")))
    except Exception:
        raise PairedProbeRefused("H334 PAIRED_EXPOSED_PROBE_REFUSED") from None
    return {
        "status": STATUS,
        "original_source_html_git_blob": BLOB,
        "paired_exposed_cases": 12,
        "arm_runs": 24,
        "aggregate": totals,
        "by_language": by_language,
        "pretrained_model_identity_attested": False,
        "free_gpu_quota_attested": False,
        "official_train_dev_admitted": False,
        "source_disjoint_blind_ood": False,
        "adapter_trained_or_gain_proven": False,
        "can_release": False,
    }

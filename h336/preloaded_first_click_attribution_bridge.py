"""H336: bridge the genuine preloaded visual policy to first-click attribution.

Source-only engineering for a *future*, independently authorized owner-private
study. The policy receives ONLY a screenshot and task; source DOM and expected
station remain inside the trusted evaluator. This module neither loads model
weights nor attests that the supplied objects are Qwen, a GPU, or trained.
Only aggregate categorical data should leave an owner-private evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from h319.beaconboard_dynamic_research import SCENES, checked_source, exact_task
from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
from h331.click_prompt_candidate import CONTROL, VARIANTS
from h335.original_first_click_error_attribution import (
    CATEGORIES, ROLE as SOURCE_ROLE, aggregate_first_click_categories,
    source_first_click,
)

ROLE = "H336_EXPOSED_PRELOADED_FIRST_CLICK_ATTRIBUTION_NOT_MODEL_ATTESTATION"
_ROW_KEYS = frozenset({
    "status", "language", "category", "source_public_exposed",
    "native_actions_dispatched", "genuine_model_identity_attested",
    "official_train_admitted", "fresh_blind_ood",
    "adapted_model_gain_proven", "can_release",
})
_COUNTER_KEYS = frozenset({
    "model_generations_via_wrapped_generate",
    "model_time_token_filters_installed", "distinct_original_pngs_observed",
    "focus_clicks", "type_calls", "abstained", "protocol_terminal",
    "protocol_failed_terminal", "pretrained_model_identity_attested",
    "real_owner_private_gpu_run_attested", "native_source_task_success_attested",
    "official_train_dev_admitted", "blind_ood_proven",
    "generalizable_hala_gain_proven",
})
_UNATTESTED = (
    "pretrained_model_identity_attested", "real_owner_private_gpu_run_attested",
    "native_source_task_success_attested", "official_train_dev_admitted",
    "blind_ood_proven", "generalizable_hala_gain_proven",
)


class FirstClickBridgeRefused(ValueError):
    """No private model output, prompt, point or source gold in exceptions."""


def need(condition: bool) -> None:
    if not condition:
        raise FirstClickBridgeRefused("H336 FIRST_CLICK_BRIDGE_REFUSED")


def verify_first_click_wrapper(row: object, counters: object) -> dict:
    """Strict one-generation/one-native-action reconciliation.

    Counter values are instrumentation claims, NOT independent model identity
    or hardware attestations. Refuse private extra fields and impossible
    transitions rather than accidentally promoting synthetic callbacks.
    """
    need(type(row) is dict and set(row) == _ROW_KEYS
         and type(counters) is dict and set(counters) == _COUNTER_KEYS)
    need(row["status"] == SOURCE_ROLE
         and type(row["language"]) is str and row["language"] in ("en", "ar")
         and type(row["category"]) is str and row["category"] in CATEGORIES
         and row["source_public_exposed"] is True
         and all(row[key] is False for key in (
             "genuine_model_identity_attested", "official_train_admitted",
             "fresh_blind_ood", "adapted_model_gain_proven", "can_release")))
    click = row["category"] != "abstain"
    need(type(row["native_actions_dispatched"]) is int
         and row["native_actions_dispatched"] == int(click))
    for key in ("model_generations_via_wrapped_generate",
                "model_time_token_filters_installed",
                "distinct_original_pngs_observed"):
        need(type(counters[key]) is int and counters[key] == 1)
    need(type(counters["focus_clicks"]) is int
         and counters["focus_clicks"] == int(click)
         and type(counters["type_calls"]) is int
         and counters["type_calls"] == 0)
    for key in ("abstained", "protocol_terminal",
                "protocol_failed_terminal", *_UNATTESTED):
        need(type(counters[key]) is bool)
    need(counters["abstained"] is (not click)
         and counters["protocol_terminal"] is (not click)
         and counters["protocol_failed_terminal"] is False
         and all(counters[key] is False for key in _UNATTESTED))
    # Only H335's safe, fixed-key row survives. No raw prediction or point.
    return dict(row)


def run_preloaded_first_click_source(
    browser: Any, source_path: Path, processor: Any, model: Any,
    scene: tuple, *, prompt_variant: str = CONTROL,
) -> dict:
    """One screenshot-conditioned model-time first click, then source oracle.

    This callable is NOT authorization to run a model or submit GPU work.
    The caller must separately preflight legal rights, pinned model revision,
    source SHA, privacy, prospective protocol and free quota.
    """
    need(type(scene) is tuple and scene in SCENES
         and type(prompt_variant) is str and prompt_variant in VARIANTS)
    try:
        # The H335 evaluator alone knows target, DOM, source focus and state.
        if prompt_variant == CONTROL:
            policy = PreloadedDynamicVisualPolicy(
                processor, model, exact_task(scene))
        else:
            policy = PreloadedDynamicVisualPolicy(
                processor, model, exact_task(scene),
                prompt_variant=prompt_variant)
        row = source_first_click(
            browser, source_path, scene, policy, role=SOURCE_ROLE)
        return verify_first_click_wrapper(
            row, policy.safe_contract_summary())
    except Exception:
        raise FirstClickBridgeRefused(
            "H336 FIRST_CLICK_SOURCE_OR_MODEL_REFUSED") from None


def run_preloaded_first_click_cohort(
    browser: Any, source_path: Path, processor: Any, model: Any,
    *, prompt_variant: str = CONTROL,
) -> dict:
    """Return only 6EN/6AR categorical totals; no per-scene data.

    Twelve source-exposed cases are not blind OOD, TRAIN admission, a
    paired adapter comparison, or a model-quality estimate.
    """
    need(type(prompt_variant) is str and prompt_variant in VARIANTS)
    try:
        checked_source(source_path)
        rows = [
            run_preloaded_first_click_source(
                browser, source_path, processor, model, scene,
                prompt_variant=prompt_variant)
            for scene in SCENES
        ]
        aggregate = aggregate_first_click_categories(rows)
        need(aggregate["original_source_scenes"] == 12
             and all(aggregate["by_language"][lang]["scenes"] == 6
                     for lang in ("en", "ar")))
    except Exception:
        raise FirstClickBridgeRefused(
            "H336 FIRST_CLICK_COHORT_REFUSED") from None
    return {
        "status": ROLE,
        "source_exposed_scenes": 12,
        "wrapped_model_time_generations_claimed": 12,
        "by_language": aggregate["by_language"],
        "source_public_exposed_not_blind": True,
        "genuine_model_identity_independently_attested": False,
        "free_gpu_quota_attested": False,
        "official_train_admitted": False,
        "adapted_gain_proven": False,
        "can_release": False,
    }

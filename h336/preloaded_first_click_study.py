"""H336 genuine preloaded VLM first-click source attribution (prospective).

Calls an existing image-only H323 model policy with the existing H319 task
on each of 12 PUBLIC EXPOSED original screenshots; H335 source oracle sees
the result only after native click. No model action, coordinates, gold, task
text, image or per-case category is emitted outside this trusted function.

This function can be mocked; its counts are NEVER evidence of genuine Qwen
until external model/kernel weights and code versions are authenticated.
No training, independent blind OOD, or learned gain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from h319.beaconboard_dynamic_research import SCENES, exact_task, checked_source
from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
from h335.original_first_click_error_attribution import (
    ROLE as SOURCE_ROLE, source_first_click,
    aggregate_first_click_categories, CATEGORIES,
)

ROLE = "H336_OWNER_PRIVATE_PRETRAINED_EXPOSED_FIRST_CLICK_DIAGNOSTIC"
ORDER = (0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11)


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H336 MODEL_FIRST_CLICK_SOURCE_REFUSED")


def verify_first_turn(summary: object, source_row: object) -> None:
    need(type(summary) is dict and set(summary) == {
        "model_generations_via_wrapped_generate",
        "model_time_token_filters_installed",
        "distinct_original_pngs_observed",
        "focus_clicks", "type_calls", "abstained",
        "protocol_terminal", "protocol_failed_terminal",
        "pretrained_model_identity_attested",
        "real_owner_private_gpu_run_attested",
        "native_source_task_success_attested",
        "official_train_dev_admitted", "blind_ood_proven",
        "generalizable_hala_gain_proven",
    })
    need(type(source_row) is dict and source_row.get("status") == SOURCE_ROLE
         and source_row.get("category") in CATEGORIES
         and type(source_row.get("native_actions_dispatched")) is int)
    abstain = source_row["category"] == "abstain"
    need(summary["model_generations_via_wrapped_generate"] == 1
         and type(summary["model_generations_via_wrapped_generate"]) is int
         and summary["model_time_token_filters_installed"] == 1
         and type(summary["model_time_token_filters_installed"]) is int
         and summary["distinct_original_pngs_observed"] == 1
         and type(summary["distinct_original_pngs_observed"]) is int
         and type(summary["focus_clicks"]) is int
         and summary["focus_clicks"] == (0 if abstain else 1)
         and type(summary["type_calls"]) is int and summary["type_calls"] == 0
         and summary["abstained"] is abstain
         and summary["protocol_terminal"] is abstain
         and summary["protocol_failed_terminal"] is False
         and source_row["native_actions_dispatched"] == (0 if abstain else 1)
         and all(summary[key] is False for key in (
             "pretrained_model_identity_attested",
             "real_owner_private_gpu_run_attested",
             "native_source_task_success_attested",
             "official_train_dev_admitted", "blind_ood_proven",
             "generalizable_hala_gain_proven",
         )))


def run_preloaded_first_click_study(
    browser: Any, source: Path, processor: Any, model: Any,
    *, role: str, policy_factory: Callable[..., Any] = PreloadedDynamicVisualPolicy,
) -> dict:
    need(type(role) is str and role == ROLE
         and callable(policy_factory) and isinstance(source, Path))
    checked_source(source)
    need(len(SCENES) == 12 and set(ORDER) == set(range(12))
         and tuple(SCENES[i][1] for i in ORDER) == ("en", "ar") * 6)
    rows = []
    try:
        for index in ORDER:
            scene = SCENES[index]
            policy = policy_factory(processor, model, exact_task(scene))
            # The policy sees H319 source_task_prompt and its screenshot,
            # never the source scene tuple, DOM, state or original gold.
            row = source_first_click(
                browser, source, scene, policy, role=SOURCE_ROLE)
            verify_first_turn(policy.safe_contract_summary(), row)
            rows.append(row)
        summary = aggregate_first_click_categories(rows)
        need(summary["status"] == SOURCE_ROLE and
             summary["original_source_scenes"] == 12
             and summary["source_public_exposed_not_blind"] is True
             and summary["real_qwen_model_generations_attested"] is False)
    except Exception:
        raise ValueError("H336 MODEL_FIRST_CLICK_SOURCE_REFUSED") from None
    return {
        "status": ROLE,
        "frozen_exposed_scenes": 12,
        "claimed_preloaded_model_generations": 12,
        "claimed_native_clicks":
            summary["by_language"]["en"]["native_actions"]
            + summary["by_language"]["ar"]["native_actions"],
        "by_language": summary["by_language"],
        "real_model_identity_requires_external_attestation": True,
        "official_train_admitted": False,
        "fresh_blind_ood": False,
        "learned_adapter_gain": False,
        "can_release": False,
    }

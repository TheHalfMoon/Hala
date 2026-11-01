"""H323: connect an already-preloaded visual policy to original BeaconBoard.

This is source-only model integration; does not load a model, attest Qwen,
train anything, authorize dataset admission, or evaluate blind OOD.
The H319 oracle alone checks target after genuine native browser dispatch;
model callback receives only screenshots and user-visible task text.
"""
from __future__ import annotations

from typing import Any

from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
from h319.beaconboard_dynamic_research import (
    ROLE, SCENES, exact_task, run_original_dynamic_sequence,
)


class DynamicModelBridgeRefused(ValueError):
    """Fixed failure, never includes private images, gold or model text."""


_ALLOWED_FAILURES = frozenset({
    "none", "abstain", "wrong_focus_or_state",
    "wrong_code", "source_state_mismatch",
})


def need(ok: bool) -> None:
    if not ok:
        raise DynamicModelBridgeRefused("H323 PRELOADED_DYNAMIC_CONTRACT_REFUSED")


def verify_counts(result: object, counters: object) -> dict:
    """Check that wrapped model-time generations match native actions exactly.

    Does NOT attest model identity. Refuse impossible state transitions and
    never return a raw source action, predicted coordinate or task label.
    """
    need(type(result) is dict and type(counters) is dict)
    need(result.get("role") == ROLE
         and type(result.get("actions_observed")) is int
         and type(result.get("native_dispatched_actions")) is int
         and type(result.get("original_four_station_task_success")) is bool
         and type(result.get("source_state_checked")) is bool
         and result["source_state_checked"] is True
         and result.get("model_generations_proven") is False
         and result.get("official_train_admitted") is False
         and result.get("independent_blind_ood") is False
         and result.get("adapted_gain_proven") is False
         and result.get("can_release") is False
         and type(result.get("categorical_failure")) is str
         and result["categorical_failure"] in _ALLOWED_FAILURES)
    n = result["actions_observed"]
    dispatched = result["native_dispatched_actions"]
    success = result["original_four_station_task_success"]
    why = result["categorical_failure"]
    need(n in (1, 2) and dispatched in (0, 1, 2))
    for key in ("model_generations_via_wrapped_generate",
                "model_time_token_filters_installed",
                "distinct_original_pngs_observed",
                "focus_clicks", "type_calls"):
        need(type(counters.get(key)) is int)
    for key in ("abstained", "protocol_terminal",
                "protocol_failed_terminal",
                "pretrained_model_identity_attested",
                "real_owner_private_gpu_run_attested",
                "native_source_task_success_attested",
                "official_train_dev_admitted", "blind_ood_proven",
                "generalizable_hala_gain_proven"):
        need(type(counters.get(key)) is bool)
    need(not any(counters[key] for key in (
        "pretrained_model_identity_attested", "real_owner_private_gpu_run_attested",
        "native_source_task_success_attested", "official_train_dev_admitted",
        "blind_ood_proven", "generalizable_hala_gain_proven")))
    need(all(counters[key] == n for key in (
        "model_generations_via_wrapped_generate",
        "model_time_token_filters_installed",
        "distinct_original_pngs_observed")))
    focus = counters["focus_clicks"]
    typed = counters["type_calls"]
    abstain = counters["abstained"]
    terminal = counters["protocol_terminal"]
    failed = counters["protocol_failed_terminal"]
    need(focus in (0, 1) and typed in (0, 1)
         and typed <= focus and dispatched <= n)
    if n == 1:
        need(typed == 0
             and ((focus == 0 and abstain and terminal and not failed
                   and dispatched == 0 and why == "abstain" and not success)
                  or (focus == 1 and dispatched == 1 and not abstain
                      and not terminal and failed
                      and why == "wrong_focus_or_state" and not success)))
    else:
        need(focus == 1 and terminal and dispatched >= 1
             and ((abstain and typed == 0 and dispatched == 1
                   and why == "abstain" and not success)
                  or (not abstain and typed == 1
                      and ((dispatched == 1 and why == "wrong_code"
                            and not success)
                           or (dispatched == 2
                               and ((success and why == "none")
                                    or (not success
                                        and why == "source_state_mismatch")))))))
    need(not success or (n == dispatched == 2 and not abstain and not failed))
    return {
        "status": "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY",
        "source_original_task_success": success,
        "real_model_generations_claimed_by_wrapper": n,
        "native_actions_dispatched": dispatched,
        "source_categorical_failure": why,
        "pretrained_model_identity_independently_attested": False,
        "source_public_exposed": True,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "trained_model_gain_proven": False,
        "can_release": False,
    }


def run_preloaded_dynamic_source_research(
    browser: Any, source: Any, processor: Any, model: Any, scene: tuple,
) -> dict:
    """Owner-private caller must attest pinned model/revision/GPU separately."""
    need(type(scene) is tuple and scene in SCENES)
    policy = PreloadedDynamicVisualPolicy(processor, model, exact_task(scene))
    try:
        result = run_original_dynamic_sequence(
            browser, source, scene, policy, role=ROLE)
        if result["categorical_failure"] == "wrong_focus_or_state":
            policy.seal_source_focus_refusal()
        elif result["categorical_failure"] == "wrong_code":
            policy.seal_source_write_refusal()
        return verify_counts(result, policy.safe_contract_summary())
    except Exception:
        raise DynamicModelBridgeRefused(
            "H323 PRELOADED_DYNAMIC_SOURCE_OR_POLICY_FAILED"
        ) from None

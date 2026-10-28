"""Isolated real original-browser actuator smoke; NO model evaluation.

Uses *source-native reference points only inside this ephemeral test* to
exercise the same click execution and reward path later used by H173.
The points are NEVER saved, sent to Kaggle, or called model predictions.
No owner-private media or scored/consumed OOD is opened here.
"""
import argparse
import json
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import serve
from h164.native_action_contract import parse_action
from h173.capture_ood import original_native_scene
from h173.source_preflight import preregister_provisional_cases, verified_original_source
from h173.stable_capture import same_original_pixels

# Two modal styles, both carousel directions and distinct slider layouts.
# This QA is deliberately a small contract smoke, NOT a fresh model OOD.
INDICES = (0, 1, 16, 17, 24, 25)


def checked_click(point):
    raw = json.dumps({"actions": [{"kind": "click", "point": point}]})
    proposal = parse_action(raw)
    if proposal.get("valid") is not True or proposal.get("kind") != "click":
        raise ValueError("H173 native actuator fixture failed typed click contract")
    return proposal


def run(upstream):
    from playwright.sync_api import sync_playwright

    upstream = Path(upstream)
    verified_original_source(upstream)
    specs = preregister_provisional_cases()
    count_correct = 0
    count_abstain = 0
    with serve(upstream) as url:
        with sync_playwright() as playwright:
            opts = {"headless": True}
            chromium = _chromium_binary()
            if chromium:
                opts["executable_path"] = chromium
            browser = playwright.chromium.launch(**opts)
            try:
                for index in INDICES:
                    spec = specs[index]
                    # Source-native reference QA is never mixed with the
                    # private owner's one-shot model OOD scoring stream.
                    frozen, instruction, point, ref_state = original_native_scene(
                        browser, url, spec, execute_reference=True
                    )
                    expected_index = ref_state.get("reference_after_index")
                    test_action = checked_click(point)
                    render, task, privileged_point, state, result = original_native_scene(
                        browser, url, spec, execute_reference=False,
                        model_action=test_action, frozen_image=frozen,
                        expected_after_index=expected_index,
                    )
                    if (
                        privileged_point is not None
                        or not same_original_pixels(render, frozen)
                        or task != instruction
                        or state["kind"] != ref_state["kind"]
                        or state["value"] != ref_state["value"]
                        or result["success"] is not True
                        or result["valid_click"] is not True
                    ):
                        raise ValueError(
                            "H173 known-original reference click did not score "
                            "correctly through ACTUAL model-action actuator: "
                            + str(spec["index"])
                        )
                    count_correct += 1
                    # A typed abstain is safe: no model click and no reward,
                    # even where a known source-native reference would pass.
                    abstain = parse_action('{"actions":[{"kind":"abstain"}]}')
                    _, _, abstain_privileged_point, _, negative = original_native_scene(
                        browser, url, spec, execute_reference=False,
                        model_action=abstain, frozen_image=frozen,
                        expected_after_index=expected_index,
                    )
                    if (
                        abstain_privileged_point is not None
                        or negative["success"] is not False
                        or negative["valid_click"] is not False
                    ):
                        raise ValueError("H173 abstention incorrectly counted as browser success")
                    count_abstain += 1
                    print(
                        "H173_EPHEMERAL_NATIVE_ACTUATOR_QA",
                        spec["app"], spec["index"],
                        "click_pass", "abstain_no_reward", flush=True
                    )
            finally:
                browser.close()
    if count_correct != len(INDICES) or count_abstain != len(INDICES):
        raise ValueError("H173 independent original actuator smoke incomplete")
    return {
        "status": "H173_PRISTINE_ORIGINAL_NATIVE_ACTUATOR_SOURCE_ONLY_PASS",
        "actual_source_native_click_reward_matches": count_correct,
        "typed_abstain_no_reward_matches": count_abstain,
        "owner_private_ood_evaluations": 0,
        "model_gpu_calls": 0,
        "optimizer_steps": 0,
        "model_quality_measured": False,
        "reference_points_persisted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream), indent=2, sort_keys=True))

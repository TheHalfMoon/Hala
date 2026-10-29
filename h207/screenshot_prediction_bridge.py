"""H207: connect model-visible original GUI screenshot to guarded native action.

Only an INJECTED caller-provided policy adapter: NOT a model, trained
checkpoint, accuracy proof, model API, or official dataset admission.
Never sees DOM, page object, selectors, private gold/reference or old OOD.
Copy-first H199 prompt and H200 strict action parser with H203 screenshot
attestation and H202 native browser transition guards.
"""
from __future__ import annotations

from typing import Any, Callable

from h199.native_action_sft import parse_actions, source_task_prompt
from h201.original_native_action_executor import (
    _bounded_preaction_png_digest,
    execute_attested_native_action,
    require_registered_page,
)

# The predictor receives exactly two pieces of permitted input: PNG
# screenshot bytes and bounded task prompt. Never a page/context handle.
ScreenshotPolicy = Callable[[bytes, str], str]


def run_screenshot_only_policy(
    page: Any,
    user_task: str,
    registered_url: str,
    predict: ScreenshotPolicy,
) -> dict:
    """Run a screenshot-only policy then verify *that* PNG before action.

    Do not pass DOM/HTML/locator labels, hidden app state, source gold
    or browser objects to the policy. Model owners must separately attest
    processor input identity and approved model/data provenance.
    """
    # Fail before screenshot or inference if the requested task/policy or
    # page isn't safe to use. No arbitrary external webpage execution.
    require_registered_page(page, registered_url)
    prompt = source_task_prompt(user_task)
    if not callable(predict):
        raise ValueError("H207 requires a screenshot-only predictor callable")

    # Capture ONCE: the exact immutable bytes we send to the predictor
    # are the bytes later compared to a fresh H203 preaction screenshot.
    visible_png = page.screenshot(animations="disabled", full_page=False)
    _bounded_preaction_png_digest(visible_png)
    require_registered_page(page, registered_url)

    # The adapter may abstain; exceptions and malformed responses fail
    # closed before any native event. It never receives the page object.
    raw = predict(visible_png, prompt)
    parsed = parse_actions(raw)
    require_registered_page(page, registered_url)
    result = execute_attested_native_action(
        page, raw, registered_url, visible_png,
    )
    return {
        "status": result["status"],
        "kind": result["kind"],
        "native_events": result["native_events"],
        "preaction_png_verified": result["preaction_png_verified"],
        "strict_action_json_verified": parsed["actions"][0]["kind"] == result["kind"],
        # Without an independently validated model identity and fresh
        # held-out eval, never label an injected policy as real Hala AI.
        "model_identity_attested": False,
        "training_data_admitted": False,
        "ood_evaluation_performed": False,
        "private_media_persisted": False,
    }

"""H407 testable LOCAL source-review boundary before model-driven native clicks.

This is a separate opt-in guarded path, not a replacement for H207's
explicit fixture-only research entrypoint. Only a separately supplied, trusted
source/human reviewer can attest ONE screenshot-visible target. This module
CANNOT authenticate that reviewer or infer independent OOD/real-GUI safety.
No DOM, model weights, private gold, network, GPU, or screenshot persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import json
import math
import re
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h201.original_native_action_executor import (
    _bounded_preaction_png_digest,
    execute_attested_native_action,
    require_registered_page,
)

ScreenshotPolicy = Callable[[bytes,str],str]
SourceReview = Callable[[bytes,str,dict],object]


@dataclass(frozen=True)
class VisualTargetReview:
    """Untrusted until bound/validated against current screenshot and action.

    The *caller* must independently authenticate the human/local source
    observer. A model MUST NOT act as its own reviewer. These fields alone
    are not cryptographic authority or real-world target-count proof.
    """
    png_sha256: str
    task_sha256: str
    action_sha256: str
    matching_targets: int
    visible_box: tuple[float,float,float,float]


def action_digest(action: object) -> str:
    """Canonical SHA over the single parsed native action, not mutable text."""
    if type(action) is not dict:
        raise ValueError("H407 reviewed action must be a parsed native dictionary")
    encoded=json.dumps(action,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False).encode("utf8")
    return sha256(encoded).hexdigest()


def _valid_sha(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None


def _require_review(
    proof: object, *, png: bytes, user_task: str, action: dict,
) -> None:
    """Refuse any absent/ambiguous target, wrong pixel/action/task or box."""
    if type(proof) is not VisualTargetReview:
        raise ValueError("H407 source review absent or untrusted")
    expected=(
        (proof.png_sha256,sha256(png).hexdigest()),
        (proof.task_sha256,sha256(user_task.encode("utf8")).hexdigest()),
        (proof.action_sha256,action_digest(action)),
    )
    if not all(_valid_sha(supplied) and compare_digest(supplied,current)
               for supplied,current in expected):
        raise ValueError("H407 source review does not bind the current frame/task/action")
    if type(proof.matching_targets) is not int or proof.matching_targets!=1:
        raise ValueError("H407 only one verified unique visual target permits a click")
    box=proof.visible_box
    if (type(box) is not tuple or len(box)!=4
        or any(type(v) not in (int,float) or not math.isfinite(v)
               for v in box)):
        raise ValueError("H407 missing or invalid visible target box")
    left,top,right,bottom=box
    if (not (0<=left<right<WIDTH and 0<=top<bottom<HEIGHT)
        or not (left<=action["point"][0]<=right and
                top<=action["point"][1]<=bottom)):
        raise ValueError("H407 proposed click misses the verified unique visible target")


def run_source_reviewed_policy(
    page: Any, user_task: str, registered_url: str,
    predict: ScreenshotPolicy, review: SourceReview,
) -> dict:
    """Research-only fail-closed screenshot->model->source review->native click.

    It uses H201/H203 exact fresh screenshot comparison at dispatch; the
    source reviewer receives only the in-memory same PNG, task, parsed action
    (not the page). It is caller's responsibility to ensure review is truly
    independent of model inference and authenticated. Not a qualified product.
    """
    require_registered_page(page, registered_url)
    prompt=source_task_prompt(user_task)
    if not callable(predict):
        raise ValueError("H407 screenshot-only model callback required")
    if not callable(review):
        raise ValueError("H407 separately authenticated source review required")
    png=page.screenshot(animations="disabled",full_page=False)
    _bounded_preaction_png_digest(png)
    require_registered_page(page,registered_url)
    raw=predict(png,prompt)
    parsed=parse_actions(raw)
    action=parsed["actions"][0]
    kind=action["kind"]
    if kind not in ("click","abstain"):
        raise ValueError("H407 click-only review does not admit other native actions")
    reviewed=False
    if kind=="click":
        before=action_digest(action)
        try:
            decision=review(png,user_task,dict(action))
        except Exception:
            # Never expose reviewer private input/error content to a model
            # or native browser workflow; decline without browser effects.
            raise ValueError("H407 independent source review declined") from None
        if action_digest(action)!=before:
            raise ValueError("H407 reviewer mutated proposed action")
        _require_review(decision,png=png,user_task=user_task,action=action)
        reviewed=True
    require_registered_page(page,registered_url)
    result=execute_attested_native_action(page,raw,registered_url,png)
    return {
        "status":result["status"],
        "kind":result["kind"],
        "native_events":result["native_events"],
        "preaction_png_verified":result["preaction_png_verified"],
        "local_review_bound_to_png":reviewed,
        # Deliberately never upgrade unauthenticated local approval into
        # genuine external OOD, independent custodian, model or IP release.
        "independent_ood_qualified":False,
        "real_gui_safety_certified":False,
        "official_model_rights_admitted":False,
        "media_persisted":False,
    }

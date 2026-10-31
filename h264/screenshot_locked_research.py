"""H264: screenshot-locked offline research loop for H262 first-party Stockroom.

A preloaded, local screenshot-only policy can be injected (e.g. H208).
Neither the model identity nor model skill is verified here; do NOT claim
inference from a stub policy, TRAIN admission, OOD or task success.
No private screenshots, output predictions or reference labels persisted.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
from typing import Callable, Any

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h201.original_native_action_executor import _bounded_preaction_png_digest
from h263.native_browser_executor import execute_one, new_offline_context

STOCKROOM_SHA256 = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
STATUS = "H264_FIRST_PARTY_SCREENSHOT_LOCKED_ACTION_RESEARCH_ONLY"
MAX_TURNS = 8


def checked_stockroom(path: Path) -> tuple[Path, str]:
    """Allow only exact H262 original first-party source BYTES, not arbitrary sites."""
    if not isinstance(path, Path):
        raise ValueError("H264 requires an explicit first-party file path")
    try:
        # Reject symlink source, including a symlinked parent, not just name.
        source = path.absolute()
        if (source.name != "stockroom.html" or source.parent.name != "h262"
                or source.is_symlink()
                or source.parent.is_symlink()
                or not source.is_file()
                or not source.resolve(strict=True).samefile(source)):
            raise ValueError("H264 first-party source path is not canonical")
        if not compare_digest(sha256(source.read_bytes()).hexdigest(), STOCKROOM_SHA256):
            raise ValueError("H264 first-party source bytes changed")
        return source, source.as_uri()
    except (OSError, RuntimeError) as exc:
        raise ValueError("H264 first-party source missing or inaccessible") from None


def _recheck(page: Any, source_path: Path, expected_url: str) -> None:
    _, url = checked_stockroom(source_path)
    if (url != expected_url or page.url != expected_url
            or page.viewport_size != {"width": WIDTH, "height": HEIGHT}):
        raise ValueError("H264 browser left pinned first-party source/viewport")


def _observed_png(page: Any) -> bytes:
    png = page.screenshot(animations="disabled", full_page=False)
    _bounded_preaction_png_digest(png)
    # Fully decode rather than trusting only 24 PNG header bytes.
    from PIL import Image
    try:
        with Image.open(BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                raise ValueError("H264 original page screenshot format or shape changed")
            image.load()
    except (ValueError, OSError):
        raise ValueError("H264 preaction screenshot pixels invalid") from None
    return png


def screenshot_locked_turn(
    page: Any,
    source_path: Path,
    expected_url: str,
    user_task: str,
    predict: Callable[[bytes, str], str],
) -> dict:
    """One preaction screenshot -> injected policy -> stale check -> ONE native action.

    Policy sees ONLY original PNG + screenshot-only task text; never page,
    DOM, source path, selectors, gold, oracle or browser state.
    """
    _recheck(page, source_path, expected_url)
    if not callable(predict):
        raise ValueError("H264 requires an injected screenshot-only policy")
    prompt = source_task_prompt(user_task)
    before = _observed_png(page)
    _recheck(page, source_path, expected_url)
    raw = predict(before, prompt)
    # Parse BEFORE attempting any action or checking DOM.
    native = parse_actions(raw)["actions"][0]
    _recheck(page, source_path, expected_url)
    refreshed = _observed_png(page)
    _recheck(page, source_path, expected_url)
    if not compare_digest(
        _bounded_preaction_png_digest(before),
        _bounded_preaction_png_digest(refreshed),
    ):
        raise ValueError("H264 stale screenshot: browser changed after model observation")
    result = execute_one(page, raw)
    _recheck(page, source_path, expected_url)
    return {"kind": result["kind"], "executed": result["executed"],
            "screenshot_locked": True, "task_success_measured": False,
            "model_identity_attested": False}


def run_local_research_episode(
    browser: Any,
    source_path: Path,
    user_task: str,
    predict: Callable[[bytes, str], str],
    *,
    max_turns: int = 1,
) -> dict:
    """Create a fresh download-disabled offline context and never persist media.

    This method is unscored and must not be used for H173 consumed scored OOD,
    user pages, external services, or arbitrary original third-party sources.
    """
    if type(max_turns) is not int or not 1 <= max_turns <= MAX_TURNS:
        raise ValueError("H264 permits only 1..8 bounded research turns")
    source, url = checked_stockroom(source_path)
    if not callable(predict):
        raise ValueError("H264 screenshot-only policy required")
    source_task_prompt(user_task)
    context = new_offline_context(browser)
    counts: Counter[str] = Counter()
    attempts = 0
    try:
        # Separate from offline emulation: fail-closed external subresource route.
        def block_external(route: Any) -> None:
            request_url = route.request.url
            if request_url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        context.route("**/*", block_external)
        page = context.new_page()
        page.goto(url, wait_until="load")
        _recheck(page, source, url)
        for _ in range(max_turns):
            decision = screenshot_locked_turn(
                page, source, url, user_task, predict)
            counts[decision["kind"]] += 1
            attempts += 1
            if not decision["executed"]:
                break
    finally:
        context.close()
    return {
        "status": STATUS,
        "policy_calls": attempts,
        "action_kind_counts": {k: counts[k] for k in sorted(counts)},
        "screenshot_stale_guard_enforced": True,
        "real_browser_task_success_measured": False,
        "real_model_inference_verified": False,
        "model_identity_attested": False,
        "private_media_written": False,
        "train_admitted": False,
        "dev_admitted": False,
        "ood_evaluated": False,
        "neural_improvement_proven": False,
    }

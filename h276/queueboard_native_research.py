"""H276: pinned first-party Queueboard screenshot-to-native-action research.

A policy receives only an in-memory screenshot and task prompt. This is
NOT model attestation, admitted TRAIN/DEV, a sealed OOD evaluation, or proof
of model skill. All source and ticket data here are invented and public.
"""
from __future__ import annotations

from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path
from typing import Any, Callable

from h199.native_action_sft import HEIGHT, WIDTH, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h264.screenshot_locked_research import _observed_png

# Public Git blob of h275/queueboard.html, not a private media hash.
QUEUEBOARD_GIT_BLOB_SHA1 = "3cf452e84a75b017905bc14b04a8571411929891"
TICKETS = frozenset({"Q-104", "Q-118", "Q-126", "Q-137", "Q-142", "Q-153"})
STATUS = "H276_FIRSTPARTY_QUEUEBOARD_NATIVE_RESEARCH_NOT_OOD"


def checked_queueboard(path: Path) -> tuple[Path, str]:
    """Refuse arbitrary pages, altered source bytes and symlinked sources."""
    if not isinstance(path, Path):
        raise ValueError("H276 requires an explicit first-party Path")
    try:
        source = path.absolute()
        if (source.name != "queueboard.html" or source.parent.name != "h275"
                or source.is_symlink() or source.parent.is_symlink()
                or not source.is_file()
                or not source.resolve(strict=True).samefile(source)):
            raise ValueError("H276 first-party source path invalid")
        raw = source.read_bytes()
        git_blob = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
        if not compare_digest(sha1(git_blob).hexdigest(), QUEUEBOARD_GIT_BLOB_SHA1):
            raise ValueError("H276 first-party source identity drift")
        return source, source.as_uri()
    except (OSError, RuntimeError):
        raise ValueError("H276 first-party source inaccessible") from None


def _recheck(page: Any, source: Path, url: str) -> None:
    _, actual = checked_queueboard(source)
    if (actual != url or page.url != url
            or page.viewport_size != {"width": WIDTH, "height": HEIGHT}):
        raise ValueError("H276 source URL or screenshot viewport changed")


def screenshot_locked_queueboard_turn(
    page: Any, source: Path, url: str, task: str,
    predict: Callable[[bytes, str], str],
) -> dict:
    """One image-conditioned policy call, strict H199 JSON, one native action.

    Neither the DOM nor a ticket reference/target point enters predict().
    The page is checked before observation, after prediction, and after action.
    """
    if not callable(predict):
        raise ValueError("H276 screenshot-only policy required")
    _recheck(page, source, url)
    prompt = source_task_prompt(task)
    before = _observed_png(page)
    _recheck(page, source, url)
    raw = predict(before, prompt)
    parse_actions(raw)
    _recheck(page, source, url)
    after = _observed_png(page)
    _recheck(page, source, url)
    if not compare_digest(sha256(before).digest(), sha256(after).digest()):
        raise ValueError("H276 stale screenshot: no native action dispatched")
    result = execute_one(page, raw)
    _recheck(page, source, url)
    return {"kind": result["kind"], "executed": result["executed"],
            "screenshot_locked": True, "model_identity_attested": False}


def _ticket_state(page: Any) -> dict[str, str]:
    """Private after-action state oracle; never called by the policy."""
    rows = page.locator('article.ticket-row[data-ticket-id]')
    found = {}
    for index in range(rows.count()):
        row = rows.nth(index)
        key = row.get_attribute("data-ticket-id")
        state = row.get_attribute("data-status")
        if key not in TICKETS or key in found or state not in ("waiting", "done"):
            raise ValueError("H276 unexpected first-party ticket state")
        found[key] = state
    return found


def run_firstparty_queueboard_probe(
    browser: Any, source_path: Path, task: str,
    target_ticket: str, predict: Callable[[bytes, str], str],
    *, language: str = "en", priority_only: bool = False,
) -> dict:
    """Unscored one-step source research, with a post-inference DOM oracle.

    The target ticket is held by the evaluator, not passed to predict().
    Success is exactly the intended row becoming done, no neighbor changed,
    and the completed counter incrementing once. It is NOT blind OOD.
    """
    if (type(target_ticket) is not str or target_ticket not in TICKETS
            or type(language) is not str or language not in ("en", "ar")
            or type(priority_only) is not bool):
        raise ValueError("H276 unsupported invented research scene")
    source, url = checked_queueboard(source_path)
    context = new_offline_context(browser)
    try:
        def local_only(route: Any) -> None:
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        context.route("**/*", local_only)
        page = context.new_page()
        page.goto(url, wait_until="load")
        if language == "ar":
            page.locator("#lang").click()
        if priority_only:
            page.locator("#priority").click()
        # A focused text field could blink while a model generates.
        page.evaluate("document.activeElement && document.activeElement.blur()")
        _recheck(page, source, url)
        if page.locator("#completed-count").inner_text() != "0":
            raise ValueError("H276 original scene is not fresh")
        if page.locator('article.ticket-row[data-ticket-id="' + target_ticket + '"]').count() != 1:
            raise ValueError("H276 intended invented row not visible")
        before = _ticket_state(page)
        if before.get(target_ticket) != "waiting" or any(v != "waiting" for v in before.values()):
            raise ValueError("H276 unexpected preaction ticket state")
        decision = screenshot_locked_queueboard_turn(page, source, url, task, predict)
        after = _ticket_state(page)
        completed = page.locator("#completed-count").inner_text()
        success = bool(decision["executed"] and decision["kind"] == "click"
                       and completed == "1" and after.get(target_ticket) == "done"
                       and all(v == ("done" if k == target_ticket else "waiting")
                               for k, v in after.items()))
        return {
            "status": STATUS, "policy_calls": 1,
            "native_action_kind": decision["kind"],
            "native_action_dispatched": decision["executed"],
            "firstparty_task_success": success,
            "firstparty_source_family": "hala-original-queueboard-h275",
            "real_model_inference_attested": False,
            "training_data_admitted": False, "independent_blind_ood": False,
            "learned_model_gain_proven": False, "private_media_saved": False,
        }
    finally:
        context.close()

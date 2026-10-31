"""H281: original bilingual ClinicGrid screenshot-to-native-action source research.

Third distinct synthetic, author-owned GUI family. This is a public source
research fixture, NOT human patient data, an approved TRAIN/DEV set, a sealed
blind OOD evaluation, pretrained model inference, or adapted-model gain.

Model callback receives only a fresh PNG and exact task text: no DOM,
target index, reference coordinates, full state or original JS.
"""
from __future__ import annotations

from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path
import re
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h264.screenshot_locked_research import _observed_png

CLINICGRID_BLOB_SHA1 = "dacdfaf5c4b9fe9ed22624109c917f7e72bf4edb"
VISITS = frozenset({"C-214", "C-308", "C-117", "C-426", "C-552", "C-633"})
STATUS = "H281_ORIGINAL_CLINICGRID_RESEARCH_NOT_OOD"
SCENES = (
    ("en_urgent_c214", "en", True, "C-214"),
    ("ar_all_c308", "ar", False, "C-308"),
)


def checked_clinicgrid(path: Path) -> tuple[Path, str]:
    """Fail closed on path alias, original GUI byte or source changes."""
    if not isinstance(path, Path):
        raise ValueError("H281 source must be a Path")
    try:
        source = path.absolute()
        if (source.name != "clinicgrid.html"
                or source.parent.name != "h281"
                or source.is_symlink() or source.parent.is_symlink()
                or not source.is_file()
                or not source.resolve(strict=True).samefile(source)):
            raise ValueError("H281 original source path invalid")
        raw = source.read_bytes()
        blob = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
        if not compare_digest(sha1(blob).hexdigest(), CLINICGRID_BLOB_SHA1):
            raise ValueError("H281 first-party source identity drift")
        return source, source.as_uri()
    except (OSError, RuntimeError):
        raise ValueError("H281 original source inaccessible") from None


def exact_task(scene: tuple) -> str:
    if scene not in SCENES:
        raise ValueError("H281 unregistered synthetic research scene")
    _, language, _, visit = scene
    if language == "ar":
        return "اعتمد الموعد الخيالي " + visit
    return "Approve fictional visit " + visit


def _verify_target(task: str, visit: str) -> None:
    if (type(task) is not str or type(visit) is not str
            or visit not in VISITS or not 8 <= len(task) <= 240
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in task)):
        raise ValueError("H281 unsupported source-owned task")
    if re.findall(r"(?<![A-Za-z0-9-])C-[0-9]{3}(?![A-Za-z0-9-])", task) != [visit]:
        raise ValueError("H281 mismatched or ambiguous visit identifier")
    if task not in ("Approve fictional visit " + visit,
                    "اعتمد الموعد الخيالي " + visit):
        raise ValueError("H281 non-preregistered task prompt")


def _recheck(page: Any, source: Path, url: str) -> None:
    _, expected = checked_clinicgrid(source)
    if (expected != url or page.url != url
            or page.viewport_size != {"width": WIDTH, "height": HEIGHT}):
        raise ValueError("H281 source path/viewport mismatch")


def _all_visits_state(page: Any) -> dict[str, bool]:
    """Evaluator-only all-six-visit state, including hidden urgent-filter rows."""
    records = page.evaluate(
        "() => visits.map(({id, done}) => ({id, done}))"
    )
    if type(records) is not list or len(records) != len(VISITS):
        raise ValueError("H281 original visit state incomplete")
    found = {}
    for row in records:
        if (type(row) is not dict or set(row) != {"id", "done"}
                or type(row["id"]) is not str
                or row["id"] not in VISITS or row["id"] in found
                or type(row["done"]) is not bool):
            raise ValueError("H281 invalid original visit state")
        found[row["id"]] = row["done"]
    if set(found) != VISITS:
        raise ValueError("H281 fictional visit identities drift")
    return found


def screenshot_locked_clinicgrid_turn(
    page: Any, source: Path, url: str, task: str,
    predict: Callable[[bytes, str], str],
) -> dict:
    """Exactly one model-style screenshot callback then one native action."""
    if not callable(predict):
        raise ValueError("H281 screenshot-only callback required")
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
        raise ValueError("H281 stale screenshot; native action forbidden")
    result = execute_one(page, raw)
    _recheck(page, source, url)
    return {"kind": result["kind"], "executed": result["executed"],
            "screenshot_locked": True, "real_model_inference_attested": False}


def run_original_clinicgrid_probe(
    browser: Any, source_path: Path, scene: tuple,
    predict: Callable[[bytes, str], str],
) -> dict:
    """Research only: independent post-click full-state source oracle.

    User task text includes visit ID, but the oracle target and DOM state
    are never passed as a separate model input. Filtered-out visits
    are checked for unchanged state; no oracle retargeting of wrong clicks.
    """
    if scene not in SCENES:
        raise ValueError("H281 scene not preregistered")
    name, lang, urgent_only, target = scene
    task = exact_task(scene)
    _verify_target(task, target)
    source, url = checked_clinicgrid(source_path)
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
        if lang == "ar":
            page.locator("#lang").click()
        if urgent_only:
            page.locator("#urgent").click()
        page.evaluate("document.activeElement && document.activeElement.blur()")
        _recheck(page, source, url)
        if page.locator("#approved-count").inner_text() not in (
            "Approved: 0", "تم الاعتماد: 0"):
            raise ValueError("H281 non-fresh original visit source")
        if page.locator('article.visit[data-visit-id="' + target + '"]').count() != 1:
            raise ValueError("H281 preregistered source target not visible")
        before = _all_visits_state(page)
        if any(before.values()):
            raise ValueError("H281 source state not freshly reset")
        decision = screenshot_locked_clinicgrid_turn(
            page, source, url, task, predict)
        after = _all_visits_state(page)
        completed = page.locator("#approved-count").inner_text()
        success = bool(
            decision["kind"] == "click" and decision["executed"]
            and completed in ("Approved: 1", "تم الاعتماد: 1")
            and after == {visit: (visit == target) for visit in VISITS}
        )
        return {
            "scene": name, "status": STATUS, "policy_calls": 1,
            "native_action_kind": decision["kind"],
            "native_action_dispatched": decision["executed"],
            "firstparty_task_success": success,
            "all_source_visits_checked": len(after),
            "real_model_inference_attested": False,
            "adapted_model_gain_proven": False,
            "training_data_admitted": False,
            "independent_blind_ood": False,
            "raw_private_media_saved": False,
        }
    finally:
        context.close()

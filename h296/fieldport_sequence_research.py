"""H296: genuinely original offline form focus->type native sequence (DEV research).

Only a rendered screenshot and the user-visible bilingual task are passed
to each policy invocation. A fixed source-owned DOM oracle runs AFTER
native action dispatch. This is PUBLIC exposed research, not real Qwen,
not official TRAIN/DEV admission, and not blind OOD.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path
import re
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h264.screenshot_locked_research import _observed_png

BLOB = "14bc9041fabf59d76adb0fc78b6a16d6d53c6e80"
ROLE = "PUBLIC_EXPOSED_ORIGINAL_FORM_SEQUENCE_DEV_RESEARCH_NOT_ADMITTED_NOT_OOD"
IDS = ("F-10", "F-20", "F-30", "F-40")
SCENES = tuple(
    (
        "field_dev_%02d" % i,
        "en" if i < 6 else "ar",
        i % 3,
        (i // 3) % 2 == 1,
        (i * 3 + i // 4) % 4,
        "FIC-%02d" % (i + 11),
    ) for i in range(12)
)
CODE = re.compile(r"FIC-[0-9]{2}\Z")


def need(value: bool, why: str) -> None:
    if not value:
        raise ValueError("H296 original form DEV research refused: " + why)


def checked_source(path: Path) -> tuple[Path, str]:
    need(isinstance(path, Path), "explicit authored GUI source Path")
    try:
        p = path.absolute()
        need(p.name == "fieldport.html" and p.parent.name == "h296"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p),
             "original source path missing or symbolic")
        raw = p.read_bytes()
        git_blob = sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
        ).hexdigest()
        need(compare_digest(BLOB, git_blob), "original GUI Git blob changed")
        return p, p.as_uri()
    except (OSError, RuntimeError):
        raise ValueError("H296 original source cannot be read") from None


def exact_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in SCENES,
         "unregistered user task or source scene")
    _, lang, _, _, index, value = scene
    need(index in range(4) and CODE.fullmatch(value) is not None,
         "invalid predefined fictional code")
    return (("Enter fictional code " + value + " into field " + IDS[index]
             + ". First select the visible field, then type the code.")
            if lang == "en" else
            ("اكتب الرمز الخيالي " + value + " في الحقل " + IDS[index]
             + ". اختر الحقل المرئي أولاً، ثم اكتب الرمز."))


def inspect_all_fields(page: Any) -> dict[str, str]:
    rows = page.evaluate("() => window.fieldPortState()")
    need(type(rows) is list and len(rows) == 4,
         "all four original form fields must exist")
    found = {}
    for item in rows:
        need(type(item) is dict and set(item) == {"id", "value"}
             and type(item["id"]) is str and item["id"] in IDS
             and item["id"] not in found and type(item["value"]) is str
             and len(item["value"]) <= 12,
             "missing/duplicate/invalid source-owned form field")
        found[item["id"]] = item["value"]
    need(set(found) == set(IDS), "all four synthetic field IDs required")
    return found


@contextmanager
def original_scene(browser: Any, source: Path, url: str, scene: tuple):
    context = new_offline_context(browser)
    try:
        def offline(route: Any) -> None:
            if route.request.url.startswith(("http:", "https:", "ws:", "wss:")):
                route.abort()
            else:
                route.continue_()
        context.route("**/*", offline)
        page = context.new_page()
        page.goto(url, wait_until="load")
        _, language, order, night, _, _ = scene
        page.evaluate(
            "options => window.configureFieldPort(options)",
            {"lang": language, "order": order, "night": night})
        need(page.url == url and page.viewport_size == {
            "width": WIDTH, "height": HEIGHT},
            "fixed real screenshot and origin required")
        need(page.locator("html").get_attribute("dir") ==
             ("rtl" if language == "ar" else "ltr"),
             "genuine bilingual direction required")
        need(page.locator("input[data-field-id]").count() == 4,
             "all original visible inputs required")
        yield page
    finally:
        context.close()


def _screen_locked_predict(page: Any, callback: Callable[[bytes, str], str],
                           task: str) -> str:
    png = _observed_png(page)
    need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n"),
         "genuine browser screenshot unavailable")
    digest = sha256(png).digest()
    url = page.url
    native = callback(png, source_task_prompt(task))
    parse_actions(native)  # no model-side forgery or multi-action result
    need(page.url == url and compare_digest(
        digest, sha256(_observed_png(page)).digest()),
        "screenshot changed during model policy computation")
    return native


def run_original_form_sequence(
    browser: Any, source_path: Path, scene: tuple,
    predict: Callable[[bytes, str], str], *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE,
         "not TRAIN, private BLIND OOD, or release")
    need(type(scene) is tuple and scene in SCENES,
         "only preregistered original DEV research cases")
    need(callable(predict), "screen-only prediction callback required")
    source, url = checked_source(source_path)
    task = exact_task(scene)
    target = IDS[scene[4]]
    with original_scene(browser, source, url, scene) as page:
        need(inspect_all_fields(page) == {id: "" for id in IDS},
             "original sequence must begin with no code entered")
        first = _screen_locked_predict(page, predict, task)
        first_outcome = execute_one(page, first)
        need(inspect_all_fields(page) == {id: "" for id in IDS},
             "first focus-step must not write or mutate input values")
        if first_outcome["kind"] == "abstain":
            return {
                "role": ROLE, "scene": scene[0], "native_actions": 1,
                "focus_click_dispatched": False, "type_dispatched": False,
                "source_task_success": False,
                "all_four_fields_checked": 4, "screen_lock_each_step": True,
                "actual_neural_generation_proven": False,
                "official_train_dev_admitted": False,
                "independent_blind_ood": False,
            }
        need(first_outcome == {"kind": "click", "executed": True},
             "first native step must be focus-click or abstain")
        second = _screen_locked_predict(page, predict, task)
        second_outcome = execute_one(page, second)
        need(second_outcome["kind"] in ("type_text", "abstain"),
             "second step must be type into focused field or abstain")
        states = inspect_all_fields(page)
        task_success = (
            second_outcome == {"kind": "type_text", "executed": True}
            and states == {id: scene[5] if id == target else "" for id in IDS}
            and page.locator("form").count() == 0
        )
        return {
            "role": ROLE, "scene": scene[0], "native_actions": 2,
            "focus_click_dispatched": True,
            "type_dispatched": bool(second_outcome["executed"]),
            "source_task_success": bool(task_success),
            "all_four_fields_checked": 4, "screen_lock_each_step": True,
            "actual_neural_generation_proven": False,
            "official_train_dev_admitted": False,
            "independent_blind_ood": False,
        }

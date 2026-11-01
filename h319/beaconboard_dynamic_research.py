"""H319: source-owned, original dynamic bilingual GUI research, NOT blind OOD.

A screenshot-only callback may perform one native button click followed by a
fresh-screen native type action. Source DOM is inspected only AFTER dispatch
for scoring and never passed to the callback.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path
from io import BytesIO
import json
import re
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context

BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
ROLE = "PUBLIC_AUTHOR_ORIGINAL_DYNAMIC_RESEARCH_NOT_ADMITTED_NOT_OOD"
IDS = ("B-11", "B-22", "B-33", "B-44")
SCENES = tuple((
    "beacon_%02d" % i, "en" if i < 6 else "ar",
    i % 3, (i // 3) % 2 == 1, (i * 5 + i // 4) % 4,
    "SIM-%02d" % (i + 21),
) for i in range(12))
CODE = re.compile(r"SIM-[0-9]{2}\Z")


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H319 PUBLIC_DYNAMIC_RESEARCH_REFUSED")



def _observed_png(page: Any) -> bytes:
    """Bounded real screenshot; no media is stored or returned in receipts."""
    from PIL import Image
    png = page.screenshot(animations="disabled", full_page=False)
    need(type(png) is bytes and 100 < len(png) < 4 * 1024 * 1024
         and png.startswith(b"\x89PNG\r\n\x1a\n"))
    try:
        with Image.open(BytesIO(png)) as image:
            need(image.format == "PNG" and image.size == (WIDTH, HEIGHT))
            image.load()
    except (OSError, ValueError):
        raise ValueError("H319 INVALID_SCREENSHOT") from None
    return png


def checked_source(path: Path) -> tuple[Path, str]:
    need(isinstance(path, Path))
    try:
        p = path.absolute()
        need(p.name == "beaconboard.html" and p.parent.name == "h319"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p))
        raw = p.read_bytes()
        git_blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                        + b"\x00" + raw).hexdigest()
        need(compare_digest(BLOB, git_blob))
        return p, p.as_uri()
    except (OSError, RuntimeError, ValueError):
        raise ValueError("H319 SOURCE_PIN_REFUSED") from None


def exact_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in SCENES)
    _, lang, _, _, index, code = scene
    need(index in range(4) and CODE.fullmatch(code) is not None)
    return (
        "Open fictional station " + IDS[index] + ", then enter code "
        + code + " in the newly revealed box. Use two separate steps."
        if lang == "en" else
        "افتح المحطة الخيالية " + IDS[index] + " ثم اكتب الرمز "
        + code + " في الحقل الذي يظهر. استخدم خطوتين منفصلتين."
    )


def inspect_all_stations(page: Any) -> dict[str, tuple[bool, str]]:
    rows = page.evaluate("() => window.beaconBoardState()")
    need(type(rows) is list and len(rows) == 4)
    found: dict[str, tuple[bool, str]] = {}
    for item in rows:
        need(type(item) is dict and set(item) == {"id", "open", "value"}
             and type(item["id"]) is str and item["id"] in IDS
             and item["id"] not in found and type(item["open"]) is bool
             and type(item["value"]) is str and len(item["value"]) <= 12)
        found[item["id"]] = (item["open"], item["value"])
    need(set(found) == set(IDS))
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
        _, language, order, dark, _, _ = scene
        page.evaluate(
            "options => window.configureBeaconBoard(options)",
            {"lang": language, "order": order, "dark": dark})
        need(page.url == url and page.viewport_size ==
             {"width": WIDTH, "height": HEIGHT})
        need(page.locator("html").get_attribute("dir") ==
             ("rtl" if language == "ar" else "ltr"))
        need(page.locator("button[data-station]").count() == 4)
        yield page
    finally:
        context.close()


def _observed_predict(page: Any, predict: Callable[[bytes, str], str],
                      task: str) -> dict:
    png = _observed_png(page)
    need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n"))
    before = sha256(png).digest()
    url = page.url
    answer = predict(png, source_task_prompt(task))
    parsed = parse_actions(answer)
    need(page.url == url and compare_digest(
        before, sha256(_observed_png(page)).digest()))
    return parsed["actions"][0]


def run_original_dynamic_sequence(
    browser: Any, source_path: Path, scene: tuple,
    predict: Callable[[bytes, str], str], *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE)
    need(type(scene) is tuple and scene in SCENES)
    need(callable(predict))
    source, url = checked_source(source_path)
    task = exact_task(scene)
    target, code = IDS[scene[4]], scene[5]
    seen_steps = 0
    dispatched = 0
    success = False
    refusal = "none"
    try:
        with original_scene(browser, source, url, scene) as page:
            need(inspect_all_stations(page) ==
                 {station: (False, "") for station in IDS})
            action = _observed_predict(page, predict, task)
            seen_steps = 1
            need(action["kind"] in ("click", "abstain"))
            if action["kind"] == "abstain":
                refusal = "abstain"
            else:
                outcome = execute_one(page, '{"actions":['
                    + json.dumps(action, separators=(",", ":"))
                    + ']}')
                need(outcome == {"kind": "click", "executed": True})
                dispatched += 1
                state = inspect_all_stations(page)
                if (state != {station: (station == target, "") for station in IDS}
                    or page.evaluate("""() => {
                       const e=document.activeElement;
                       return e instanceof HTMLInputElement
                         ? e.dataset.station : null;
                    }""") != target):
                    refusal = "wrong_focus_or_state"
                else:
                    action = _observed_predict(page, predict, task)
                    seen_steps = 2
                    need(action["kind"] in ("type_text", "abstain"))
                    if action["kind"] == "abstain":
                        refusal = "abstain"
                    elif action["text"] != code:
                        refusal = "wrong_code"
                    else:
                        focus = page.evaluate("""() => {
                            const e=document.activeElement;
                            return e instanceof HTMLInputElement
                              ? e.dataset.station : null;
                        }""")
                        need(focus == target)
                        outcome = execute_one(page, '{"actions":['
                            + __import__("json").dumps(action, separators=(",", ":"))
                            + ']}')
                        need(outcome == {"kind": "type_text", "executed": True})
                        dispatched += 1
                        success = inspect_all_stations(page) == {
                            station: (station == target, code if station == target else "")
                            for station in IDS
                        }
                        if not success:
                            refusal = "source_state_mismatch"
    except Exception:
        raise ValueError("H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED") from None
    return {
        "role": ROLE, "actions_observed": seen_steps,
        "native_dispatched_actions": dispatched,
        "original_four_station_task_success": success,
        "categorical_failure": refusal if not success else "none",
        "source_state_checked": True,
        "model_generations_proven": False, "official_train_admitted": False,
        "independent_blind_ood": False, "adapted_gain_proven": False,
        "can_release": False,
    }

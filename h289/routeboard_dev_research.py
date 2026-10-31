"""H289: separate, publicly exposed FIRSTPARTY DEV research only.

Neither Qwen performance, trained gain, formal DEV admission nor blind OOD
is claimed. This source evaluator gives the policy ONLY screenshot PNG +
an original bilingual user task; never DOM, reference/gold or coordinates.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha1, sha256
from hmac import compare_digest
from pathlib import Path
from typing import Any, Callable

from h199.native_action_sft import WIDTH, HEIGHT, parse_actions, source_task_prompt
from h263.native_browser_executor import execute_one, new_offline_context
from h264.screenshot_locked_research import _observed_png

BLOB = "6b7486384c935ff6ea856a0eefeb213a9504fdf2"
ROLE = "DEV_RESEARCH_ONLY_PUBLIC_EXPOSED_NOT_ADMITTED_NOT_BLIND_OOD"
IDS = ("R-602", "R-617", "R-625", "R-638", "R-644", "R-659")
SCENES = tuple(
    (
        "dev_%02d" % i,
        "en" if i < 6 else "ar",
        i % 3,
        i % 2 == 0,
        (i // 3) % 2 == 1,
        IDS[(i % 3) * 2] if i % 2 == 0 else IDS[(i * 5 + 1) % 6],
    ) for i in range(12)
)


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H289 original DEV-only source refused: " + why)


def checked_source(source: Path) -> tuple[Path, str]:
    need(isinstance(source, Path), "explicit original source Path")
    try:
        p = source.absolute()
        need(p.name == "routeboard.html" and p.parent.name == "h289"
             and p.is_file() and not p.is_symlink()
             and not p.parent.is_symlink()
             and p.resolve(strict=True).samefile(p),
             "original source link/path drift")
        data = p.read_bytes()
        git_blob = sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\x00" + data
        ).hexdigest()
        need(compare_digest(git_blob, BLOB),
             "original RouteBoard public HTML byte drift")
        return p, p.as_uri()
    except (OSError, RuntimeError):
        raise ValueError("H289 original GUI cannot be read") from None


def exact_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in SCENES,
         "H289 source DEV task not preregistered")
    name, lang, _, _, _, target = scene
    return (("Assign fictional route " + target) if lang == "en"
            else ("وجّه المسار الخيالي " + target))


def inspect_all_six(page: Any) -> dict[str, bool]:
    rows = page.evaluate("() => window.routeBoardState()")
    need(type(rows) is list and len(rows) == 6,
         "H289 original all-six-route oracle unavailable")
    result = {}
    for row in rows:
        need(type(row) is dict and set(row) == {"id", "routed"}
             and type(row["id"]) is str and row["id"] in IDS
             and row["id"] not in result and type(row["routed"]) is bool,
             "H289 non-original/missing/duplicate route state")
        result[row["id"]] = row["routed"]
    need(set(result) == set(IDS), "H289 all fictional routes required")
    return result


@contextmanager
def original_scene(browser: Any, path: Path, url: str, scene: tuple):
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
        _, lang, order, urgent, night, _ = scene
        page.evaluate(
            "options => window.configureRouteBoard(options)",
            {"language": lang, "sort": order, "urgentOnly": urgent,
             "night": night})
        need(page.url == url and page.viewport_size == {
            "width": WIDTH, "height": HEIGHT},
            "H289 offline original source URL/viewport drift")
        need(page.locator("html").get_attribute("dir") ==
             ("rtl" if lang == "ar" else "ltr"),
             "H289 original bilingual layout drift")
        need(page.locator("article.row").count() == (3 if urgent else 6),
             "H289 original visible-route filter mismatch")
        yield page
    finally:
        context.close()


def run_original_dev_probe(
    browser: Any, source_path: Path, scene: tuple,
    predict: Callable[[bytes, str], str], *, role: str,
) -> dict:
    need(type(role) is str and role == ROLE,
         "DEV-only: do not promote to TRAIN, blind OOD or release")
    need(type(scene) is tuple and scene in SCENES,
         "only preregistered author-owned DEV case")
    need(callable(predict), "screenshot-only policy callback required")
    source, url = checked_source(source_path)
    task = exact_task(scene)
    with original_scene(browser, source, url, scene) as page:
        before_state = inspect_all_six(page)
        need(not any(before_state.values()), "DEV must start fresh")
        target = scene[-1]
        need(page.locator(
            'article.row[data-route-id="' + target + '"] button'
        ).count() == 1, "DEV target must be visible, unique")
        png = _observed_png(page)
        need(png.startswith(b"\x89PNG\r\n\x1a\n"),
             "DEV real browser screenshot unavailable")
        snapshot = sha256(png).digest()
        native = predict(png, source_task_prompt(task))
        parsed = parse_actions(native)
        need(page.url == url and
             compare_digest(snapshot, sha256(_observed_png(page)).digest()),
             "DEV screenshot stale or altered by policy")
        outcome = execute_one(page, native)
        after_state = inspect_all_six(page)
        success = (
            outcome == {"kind": "click", "executed": True}
            and after_state == {id: id == target for id in IDS}
            and page.locator("#counter").inner_text()
            in ("Routed: 1", "تم التوجيه: 1")
        )
        return {
            "role": ROLE,
            "scene": scene[0],
            "native_kind": outcome["kind"],
            "native_action_dispatched": bool(outcome["executed"]),
            "firstparty_task_success": bool(success),
            "all_six_source_ids_checked": 6,
            "screenshot_locked": True,
            "dev_source_public_exposed": True,
            "formal_train_dev_admitted": False,
            "independent_blind_ood": False,
            "actual_model_generation_proven": False,
        }

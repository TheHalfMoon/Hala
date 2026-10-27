"""H159: freeze and replay a NEW TodoMVC source cohort before ANY model calls.

H158 was quarantined after 8/16 replay screenshots failed SHA identity.
This module never reads H158 predictions, changes its lock or scores a model.
All source screenshots and oracle points stay owner-private.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h158.original_todomvc_source import (
    COUNTS, PIN, point, source_url, tasks_for,
)

SCHEMA = "hala.h159.preregistered-stable-original-todomvc-source.v1"
ROOT = "private-h159"
FOLDER = "source-once-v1"
START_INDEX = 301
COUNT = 16


def digest(png):
    return hashlib.sha256(png).hexdigest()


def stable_png(page, attempts=7):
    """Disable animations, then demand *two byte-identical* post-update PNGs."""
    page.evaluate("() => document.fonts.ready")
    page.wait_for_timeout(200)
    prev = page.screenshot(type="png", animations="disabled")
    for _ in range(attempts):
        page.wait_for_timeout(125)
        latest = page.screenshot(type="png", animations="disabled")
        if latest == prev:
            return latest
        prev = latest
    raise ValueError("H159 unstable original-app screenshot; abort before GPU")


def scenario(browser, url, case, *, reference=False):
    ctx = browser.new_context(
        viewport={"width": 960, "height": 720},
        device_scale_factor=1,
        locale="en-US",
    )
    try:
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=20000)
        items = tasks_for(case["language"], case["index"])
        for item in items:
            field = page.locator("input.new-todo")
            field.fill(item)
            field.press("Enter")
        if page.locator("ul.todo-list li").count() != 3:
            raise ValueError("H159 original TodoMVC setup missing a todo")
        if case["task"] != "complete_item":
            page.locator("ul.todo-list li").filter(has_text=items[0]).locator(
                "input.toggle"
            ).click()
            page.wait_for_function(
                "() => document.querySelectorAll('ul.todo-list li.completed').length === 1",
                timeout=5000,
            )
        picture = stable_png(page)
        if not reference:
            return digest(picture)

        if case["task"] == "complete_item":
            target = page.locator("ul.todo-list li").filter(
                has_text=items[1]
            ).locator("input.toggle")
        elif case["task"] == "filter_active":
            target = page.locator("ul.filters a[href='#/active']")
        elif case["task"] == "clear_completed":
            target = page.locator("button.clear-completed")
        else:
            raise ValueError("H159 unregistered task type")
        x, y = point(target)
        page.mouse.click(x, y)
        if case["task"] == "complete_item":
            correct = page.locator("ul.todo-list li.completed").filter(
                has_text=items[1]
            ).count() == 1
        elif case["task"] == "filter_active":
            page.wait_for_function(
                "() => location.hash === '#/active'", timeout=3000
            )
            correct = (
                page.locator("ul.todo-list li.completed").count() == 0
                and page.locator("ul.todo-list li").count() == 2
            )
        else:
            correct = (
                page.locator("ul.todo-list li").filter(
                    has_text=items[0]
                ).count() == 0
                and page.locator("ul.todo-list li").count() == 2
            )
        if not correct:
            raise ValueError("H159 original source action did not succeed")
        return picture, [round(x, 4), round(y, 4)]
    finally:
        ctx.close()


def cases():
    result = []
    for task, groups in COUNTS.items():
        for language, size in groups.items():
            for _ in range(size):
                result.append({
                    "task": task, "language": language,
                    "index": START_INDEX + len(result),
                })
    if len(result) != COUNT:
        raise ValueError("H159 original benchmark quotas changed")
    return result


def capture(owner, upstream):
    owner = Path(owner)
    root = owner / ROOT / FOLDER
    if root.exists():
        raise FileExistsError("H159 original source is create-once")
    old = json.loads((
        owner / "private-h158" / "todomvc-source-once-v1" / "manifest.json"
    ).read_text("utf8"))
    excluded = {x["image_sha256"] for x in old["private_cases"]}
    if len(excluded) != 16:
        raise ValueError("H158 previous-source exclusion incomplete")
    url = source_url(upstream)
    from playwright.sync_api import sync_playwright
    from h158.original_todomvc_source import TOTAL

    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    frozen = []
    with sync_playwright() as pw:
        opts = {"headless": True}
        binary = _chromium_binary()
        if binary:
            opts["executable_path"] = binary
        browser = pw.chromium.launch(**opts, timeout=20000)
        try:
            for case in cases():
                png, gold = scenario(browser, url, case, reference=True)
                sha = digest(png)
                if sha in excluded or sha in {
                    x["image_sha256"] for x in frozen
                }:
                    raise ValueError("H159 reused previous or duplicate PNG")
                name = f'{case["task"]}-{case["language"]}-{case["index"]}.png'
                (root / "images" / name).write_bytes(png)
                frozen.append({
                    **case, "image": name, "image_sha256": sha,
                    "reference_point": gold, "reference_success": True,
                })
                print("H159_FRESH_ORIGINAL_REFERENCE", len(frozen), TOTAL,
                      flush=True)
        finally:
            browser.close()
    manifest = {
        "schema": SCHEMA, "upstream": "tastejs/todomvc",
        "source_sha": PIN, "view_size": [960, 720],
        "screenshot": "animations-disabled; two-identical-snapshots",
        "source_reference_success": 16, "old_H158_images_excluded": 16,
        "model_calls": 0, "optimizer_steps": 0, "private_cases": frozen,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (root / "manifest.json").write_bytes(data)
    return {"status": "H159_NEW_SOURCE_16_REFERENCE_PASS",
            "manifest_sha256": digest(data), "model_calls": 0}


def verify(owner, upstream):
    owner = Path(owner)
    root = owner / ROOT / FOLDER
    raw = (root / "manifest.json").read_bytes()
    data = json.loads(raw)
    source = data["private_cases"]
    if (data["schema"] != SCHEMA or data["source_sha"] != PIN
            or len(source) != 16
            or {x["index"] for x in source} != set(range(301, 317))
            or Counter((x["task"], x["language"]) for x in source)
            != Counter((t, lang) for t, groups in COUNTS.items()
                       for lang, size in groups.items() for _ in range(size))):
        raise ValueError("H159 frozen source manifest failed independent QA")
    if len({x["image_sha256"] for x in source}) != 16:
        raise ValueError("H159 duplicate source PNG")
    for case in source:
        if digest((root / "images" / case["image"]).read_bytes()) != case["image_sha256"]:
            raise ValueError("H159 frozen screenshot altered")
    url = source_url(upstream)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        opts = {"headless": True}
        binary = _chromium_binary()
        if binary:
            opts["executable_path"] = binary
        browser = pw.chromium.launch(**opts, timeout=20000)
        try:
            for n, case in enumerate(source, 1):
                if scenario(browser, url, case) != case["image_sha256"]:
                    raise ValueError(
                        f"H159 cross-context pixel mismatch at position {n}"
                    )
                print("H159_PIXEL_EXACT_REPLAY", n, 16, flush=True)
        finally:
            browser.close()
    receipt = {
        "status": "H159_16_OF_16_INDEPENDENT_SCREENSHOT_SHA_MATCH",
        "model_calls": 0, "source_images": 16,
        "new_upstream_training_images": 0, "h140_blind_calls": 0,
        "paid_gpu": False, "manifest_sha256": digest(raw),
    }
    report = owner / ROOT / "H159_SOURCE_REPLAY_QA.json"
    if report.exists():
        raise FileExistsError("H159 pre-GPU owner QA already recorded")
    report.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", "utf8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "verify"))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    fn = capture if args.action == "capture" else verify
    print(json.dumps(fn(args.owner, args.upstream), sort_keys=True, indent=2))

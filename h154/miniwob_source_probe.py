"""H154: pinned original MiniWoB++ reference-action qualification, NOT model evaluation."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary

PIN = "7fd85d71a4b60325c6585396ec4f48377d049838"
SCHEMA = "hala.h154.original-miniwob-source-actions.v1"
TASKS = {"click-test": "click", "enter-text": "type_submit",
         "choose-list": "select_submit"}
REPEATS = 4


def admit_upstream(root):
    root = Path(root)
    license_text = (root / "LICENSE").read_text("utf8")
    if "MIT License" not in license_text or "Copyright" not in license_text:
        raise ValueError("Original MiniWoB MIT license/copyright missing")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if head != PIN or dirty:
        raise ValueError("MiniWoB source is not pristine pinned commit")
    html = root / "miniwob" / "html"
    for path in ("core/core.js", "common/ui_utils.js",
                 *(f"miniwob/{t}.html" for t in TASKS)):
        if not (html / path).is_file():
            raise ValueError("MiniWoB required original web resource missing")
    return html / "miniwob"


def center(locator):
    bbox = locator.bounding_box(timeout=3000)
    if not bbox or bbox["width"] < 1 or bbox["height"] < 1:
        raise ValueError("Visible target is missing")
    return bbox["x"] + bbox["width"]/2, bbox["y"] + bbox["height"]/2


def source_oracle(page, task):
    """Oracle uses visible instruction and browser actions; NEVER model inputs."""
    if task == "click-test":
        page.mouse.click(*center(page.locator("#subbtn")))
    elif task == "enter-text":
        value = page.locator("#query .bold").inner_text(timeout=3000)
        if not value or len(value) > 100:
            raise ValueError("Original visible text goal absent")
        page.mouse.click(*center(page.locator("#tt")))
        page.keyboard.type(value)
        page.mouse.click(*center(page.locator("#subbtn")))
    elif task == "choose-list":
        query = page.locator("#query").inner_text(timeout=3000).strip()
        match = re.fullmatch(
            r"Select (.+) from the list and click Submit[.]", query)
        if not match:
            raise ValueError("Original list goal format changed")
        page.locator("#options").select_option(label=match.group(1))
        page.mouse.click(*center(page.locator("#area button").first))
    else:
        raise ValueError("Unknown action type")


def episode(browser, base, task, index, images):
    context = browser.new_context(
        viewport={"width": 960, "height": 720}, device_scale_factor=1,
        locale="en-US", service_workers="block")
    try:
        page = context.new_page()
        page.goto((base / f"{task}.html").resolve().as_uri(),
                  wait_until="load", timeout=20000)
        page.locator("#sync-task-cover").click(timeout=5000)
        page.wait_for_function(
            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
            timeout=5000)
        instruction = page.locator("#query").inner_text(timeout=3000).strip()
        if not instruction:
            raise ValueError("Original task has no visible instruction")
        screenshot = page.screenshot(type="png")
        name = f"{task}-{index:02d}.png"
        (images / name).write_bytes(screenshot)
        source_oracle(page, task)
        page.wait_for_function("() => WOB_DONE_GLOBAL === true", timeout=3500)
        score = page.evaluate(
            "() => ({done: WOB_DONE_GLOBAL, raw: WOB_RAW_REWARD_GLOBAL,"
            " episodes: WOB_EPISODE_ID})")
        return {
            "task": task, "action_class": TASKS[task], "case": index,
            "instruction": instruction,
            "png": name, "sha256": hashlib.sha256(screenshot).hexdigest(),
            "done": bool(score["done"]), "positive_raw_reward": score["raw"] > 0,
            "episode_count": score["episodes"],
            "reference_pass": bool(score["done"] and score["raw"] > 0
                                   and score["episodes"] == 1),
            "model_calls": 0,
        }
    finally:
        context.close()


def run(source, owner):
    from playwright.sync_api import sync_playwright
    base = admit_upstream(source)
    private = Path(owner) / "private-h154" / "source-only-miniwob-v2"
    if private.exists():
        raise FileExistsError("H154 source pilot is create-once; do not retune")
    private.mkdir(parents=True, exist_ok=False)
    (private / "ONE_SHOT_SOURCE.lock").write_text(SCHEMA, encoding="utf8")
    images = private / "screenshots"
    images.mkdir()
    rows = []
    with sync_playwright() as pw:
        options = {"headless": True}
        binary = _chromium_binary()
        if binary:
            options["executable_path"] = binary
        browser = pw.chromium.launch(**options, timeout=20000)
        try:
            for task in TASKS:
                for i in range(REPEATS):
                    row = episode(browser, base, task, i, images)
                    rows.append(row)
                    print("H154_ORIGINAL_MINIWOB_SOURCE_QA", len(rows),
                          len(TASKS)*REPEATS, task, int(row["reference_pass"]),
                          flush=True)
        finally:
            browser.close()
    counts = Counter(r["task"] for r in rows)
    unique = len({r["sha256"] for r in rows})
    if len(rows) != len(TASKS)*REPEATS or counts != {
            task: REPEATS for task in TASKS} or unique != len(rows):
        raise ValueError("H154 source QA count/unique image check failed")
    summary = {
        "schema": SCHEMA, "status": "H154_MINIWOB_SOURCE_QUALIFIED",
        "upstream_git_sha": PIN, "upstream_license": "MIT",
        "independent_original_app": True,
        "action_classes": list(TASKS.values()),
        "original_browser_episodes": len(rows),
        "fresh_unique_screenshots": unique,
        "original_reference_pass": sum(r["reference_pass"] for r in rows),
        "model_calls": 0, "optimizer_steps": 0,
        "model_accuracy_measured": False, "h140_blind_calls": 0,
        "paid_gpu": False, "release_qualified": False,
    }
    (private / "source-qa.json").write_text(
        json.dumps({"summary": summary, "private_cases": rows},
                   ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--owner-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.owner_root), indent=2, sort_keys=True))

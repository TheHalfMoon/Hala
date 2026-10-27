"""H155 create-once original MiniWoB source-only TRAIN/DEV action supervision.

This produces private labels for future FREE SFT. No neural training, no
independent BLIND usage, and never reports source-oracle as model accuracy.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream, TASKS, center, source_oracle
from h155.action_contract import action_from_dict, H155_SUPPORTED_TASKS

SCHEMA = "hala.h155.private-miniwob-source-only-train-dev-v1"
SPLITS = {"TRAIN": 16, "DEV": 4}  # per task: 48 TRAIN + 12 DEV
START_SEED = 310000


def owner_label(page, task):
    """Supervision only; not a model-visible action reference at eval."""
    if task == "click-test":
        result = [{"kind": "click", "point": list(
            center(page.locator("#subbtn")))}]
    elif task == "enter-text":
        visible_goal = page.locator("#query .bold").inner_text(timeout=3000)
        result = [
            {"kind": "click", "point": list(center(page.locator("#tt")))},
            {"kind": "type", "text": visible_goal},
            {"kind": "submit", "point": list(center(page.locator("#subbtn")))},
        ]
    elif task == "choose-list":
        goal = page.locator("#query").inner_text(timeout=3000).strip()
        match = re.fullmatch(
            r"Select (.+) from the list and click Submit[.]", goal)
        if not match:
            raise ValueError("Original visible MiniWoB list target missing")
        result = [
            {"kind": "select", "text": match.group(1)},
            {"kind": "submit", "point": list(
                center(page.locator("#area button").first))},
        ]
    else:
        raise ValueError("Unregistered source-supervision task")
    checked = tuple(action_from_dict(item).kind for item in result)
    if checked != H155_SUPPORTED_TASKS[task]:
        raise ValueError("Supervision violates preregistered typed action sequence")
    return {"actions": result}


def excluded_images(owner):
    root = Path(owner) / "private-h154"
    found = set()
    for name in ("source-only-miniwob-v1", "source-only-miniwob-v2"):
        previous = root / name / "screenshots"
        if not previous.is_dir():
            raise ValueError("Prior H154 pilot source SHA exclusions missing")
        for path in previous.glob("*.png"):
            found.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return found


def make_episode(browser, html_root, task, split, index, seed, output):
    ctx = browser.new_context(
        viewport={"width": 960, "height": 720},
        device_scale_factor=1, locale="en-US", service_workers="block")
    try:
        page = ctx.new_page()
        page.goto((html_root / f"{task}.html").resolve().as_uri(),
                  wait_until="load", timeout=20000)
        page.evaluate("(seed) => Math.seedrandom(String(seed))", seed)
        page.locator("#sync-task-cover").click(timeout=5000)
        page.wait_for_function(
            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
            timeout=5000)
        instruction = page.locator("#query").inner_text(timeout=3000).strip()
        if not instruction:
            raise ValueError("MiniWoB original instruction missing")
        picture = page.screenshot(type="png")
        filename = f"{split.lower()}-{task}-{index:03d}.png"
        (output / filename).write_bytes(picture)
        sha = hashlib.sha256(picture).hexdigest()
        # Oracle label used for TRAIN/DEV only, strictly never model input.
        label = owner_label(page, task)
        source_oracle(page, task)
        page.wait_for_function("() => WOB_DONE_GLOBAL === true", timeout=3500)
        true_reward = page.evaluate("() => WOB_RAW_REWARD_GLOBAL")
        if true_reward <= 0:
            raise ValueError("Original MiniWoB reference failed to complete task")
        return {
            "split": split, "task": task, "action_family": TASKS[task],
            "seed": seed, "image": filename, "image_sha256": sha,
            "instruction": instruction,
            "training_supervision": label,
            "reference_reward_positive": True,
            "model_inference_calls": 0,
        }
    finally:
        ctx.close()


def collect(source, owner):
    owner = Path(owner)
    app = admit_upstream(source)
    folder = owner / "private-h155" / "source-train-dev-v1"
    if folder.exists():
        raise FileExistsError("H155 source TRAIN/DEV corpus is create-only")
    from playwright.sync_api import sync_playwright
    excluded = excluded_images(owner)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "ONE_SHOT_SOURCE_LOCK").write_text(SCHEMA, encoding="utf8")
    media = folder / "images"
    media.mkdir()
    cases = []
    with sync_playwright() as pw:
        options = {"headless": True}
        binary = _chromium_binary()
        if binary:
            options["executable_path"] = binary
        browser = pw.chromium.launch(**options, timeout=20000)
        try:
            for split, quota in SPLITS.items():
                for task_index, task in enumerate(TASKS):
                    for i in range(quota):
                        seed = (START_SEED +
                                (0 if split == "TRAIN" else 10000) +
                                200*task_index + i)
                        row = make_episode(
                            browser, app, task, split, i, seed, media)
                        if row["image_sha256"] in excluded:
                            raise ValueError("H155 source overlaps historic H154 screenshot")
                        excluded.add(row["image_sha256"])
                        cases.append(row)
                        if len(cases) % 12 == 0:
                            print("H155_PRIVATE_SOURCE_LABELS", len(cases),
                                  60, flush=True)
        finally:
            browser.close()
    grouped = Counter((row["split"],row["task"]) for row in cases)
    if (len(cases) != 60 or grouped != {
            (split, task): count for split,count in SPLITS.items()
            for task in TASKS}):
        raise ValueError("H155 source TRAIN/DEV balance and unique SHA failed")
    manifest = {
        "schema": SCHEMA, "status": "PRIVATE_ACTION_TRAIN_DEV_SOURCE_COMPLETE",
        "original_app_git_sha": "7fd85d71a4b60325c6585396ec4f48377d049838",
        "mit_source_preserved": True,
        "train": 48, "dev": 12, "new_image_count": 60,
        "action_families": list(TASKS.values()),
        "prior_source_sha_excluded": True,
        "dev_is_not_blind": True, "model_inference_calls": 0,
        "optimizer_steps": 0, "model_accuracy_measured": False,
        "h140_blind_neural_calls": 0, "Hala_35B_unchanged": True,
        "release_qualified": False, "private_cases": cases,
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf8")
    return {k: v for k,v in manifest.items() if k != "private_cases"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--owner-root", required=True)
    opts = parser.parse_args()
    print(json.dumps(collect(opts.source, opts.owner_root),
                     indent=2, sort_keys=True))

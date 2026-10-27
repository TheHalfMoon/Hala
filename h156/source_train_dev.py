"""H156 source-only MiniWoB supervision v2; real native mouse/keyboard.

Source oracle reads DOM only to label TRAIN/DEV; actual validation replays
the serialized labels with DOM-free mouse+keyboard controller. New SHA-
disjoint cohort; no model call, no GPU, no H140 sealed blind.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream, TASKS,center
from h156.action_contract_v2 import action_from_dict_v2, native_mouse_keyboard

SCHEMA="hala.h156.private-action-train-dev-v2.v1"
SPLITS={"TRAIN":16,"DEV":4}
START_SEED=430000
APP_SHA="7fd85d71a4b60325c6585396ec4f48377d049838"


def excluded_images(owner):
    owner=Path(owner)
    sources=(
        owner/"private-h154"/"source-only-miniwob-v1"/"screenshots",
        owner/"private-h154"/"source-only-miniwob-v2"/"screenshots",
        owner/"private-h155"/"source-train-dev-v1"/"images",
    )
    hashes=set()
    for folder in sources:
        if not folder.is_dir():
            raise ValueError("Missing frozen prior source for leakage exclusion")
        for img in folder.glob("*.png"):
            hashes.add(hashlib.sha256(img.read_bytes()).hexdigest())
    if len(hashes)<81:
        raise ValueError("Historical MiniWoB SHA baseline incomplete")
    return hashes


def owner_label_v2(page,task):
    """Ground truth from original app DOM for TRAIN/DEV only."""
    if task=="click-test":
        actions=[{"kind":"click","point":list(center(page.locator("#subbtn")))}]
    elif task=="enter-text":
        value=page.locator("#query .bold").inner_text(timeout=3000)
        actions=[
            {"kind":"click","point":list(center(page.locator("#tt")))},
            {"kind":"type","text":value},
            {"kind":"submit","point":list(center(page.locator("#subbtn")))}
        ]
    elif task=="choose-list":
        instruction=page.locator("#query").inner_text(timeout=3000).strip()
        match=re.fullmatch(r"Select (.+) from the list and click Submit[.]",instruction)
        if not match:
            raise ValueError("Visible original dropdown instruction drift")
        actions=[
            {"kind":"select","point":list(center(page.locator("#options"))),
             "text":match.group(1)},
            {"kind":"submit","point":list(center(page.locator("#area button").first))}
        ]
    else:
        raise ValueError("Unknown original MiniWoB source task")
    return {"actions":actions}


def native_trial(browser,base,task,split,index,seed,media):
    context=browser.new_context(
        viewport={"width":960,"height":720},
        device_scale_factor=1,locale="en-US",service_workers="block")
    try:
        page=context.new_page()
        page.goto((base/f"{task}.html").resolve().as_uri(),
                  wait_until="load",timeout=20000)
        page.evaluate("(seed)=>Math.seedrandom(String(seed))",seed)
        page.locator("#sync-task-cover").click(timeout=5000)
        page.wait_for_function(
            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
            timeout=5000)
        instruction=page.locator("#query").inner_text(timeout=3000).strip()
        screenshot=page.screenshot(type="png")
        sha=hashlib.sha256(screenshot).hexdigest()
        filename=f"{split.lower()}-{task}-{index:03d}.png"
        (media/filename).write_bytes(screenshot)
        labels=owner_label_v2(page,task)
        actions=tuple(action_from_dict_v2(x) for x in labels["actions"])
        native=native_mouse_keyboard(page,actions)
        page.wait_for_function("() => WOB_DONE_GLOBAL === true",timeout=5000)
        reward=page.evaluate("() => WOB_RAW_REWARD_GLOBAL")
        if reward<=0 or native["status"]!="ACTIONS_SENT_NATIVE":
            raise ValueError("H156 original task not complete using DOM-free replay")
        return {
            "split":split,"task":task,"action_family":TASKS[task],
            "seed":seed,"image":filename,"image_sha256":sha,
            "instruction":instruction,"training_supervision":labels,
            "native_reference_actions_completed":True,
            "native_controller_reads_dom":False,
            "model_inference_calls":0,
        }
    finally:
        context.close()


def collect(source,owner):
    owner=Path(owner)
    base=admit_upstream(source)
    private=owner/"private-h156"/"miniwob-action-train-dev-v2"
    if private.exists():
        raise FileExistsError("H156 source is create-once; never retest or overwrite")
    excluded=excluded_images(owner)
    from playwright.sync_api import sync_playwright
    private.mkdir(parents=True,exist_ok=False)
    (private/"ONE_SHOT_ORIGINAL_SOURCE.lock").write_text(SCHEMA,encoding="utf8")
    images=private/"images"
    images.mkdir()
    rows=[]
    with sync_playwright() as playwright:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:
            opts["executable_path"]=exe
        browser=playwright.chromium.launch(**opts,timeout=20000)
        try:
            for split,num in SPLITS.items():
                for task_id,task in enumerate(TASKS):
                    for index in range(num):
                        seed=(START_SEED + 10000*(split=="DEV") +
                              200*task_id + index)
                        record=native_trial(browser,base,task,split,index,
                                            seed,images)
                        if record["image_sha256"] in excluded:
                            raise ValueError("H156 source image overlaps previous data")
                        excluded.add(record["image_sha256"])
                        rows.append(record)
                        if len(rows)%12==0:
                            print("H156_NEW_ORIGINAL_NATIVE_ACTION_LABELS",
                                  len(rows),60,flush=True)
        finally:
            browser.close()
    strata=Counter((r["split"],r["task"]) for r in rows)
    expected={(s,t):n for s,n in SPLITS.items() for t in TASKS}
    if strata!=expected or len(rows)!=60:
        raise ValueError("Original H156 48/12 train/dev strata incomplete")
    doc={
        "schema":SCHEMA,"status":"H156_NEW_ORIGINAL_EXECUTABLE_TRAIN_DEV_READY",
        "source_app_git_sha":APP_SHA,"source_app_license":"MIT",
        "train":48,"dev":12,
        "fresh_original_screenshots":len(rows),
        "excluded_history_screenshots":len(excluded)-len(rows),
        "original_native_reference_success":60,
        "train_and_dev_disjoint":True,"sealed_blind_eligible":False,
        "training_eligible":True,"model_accuracy_measured":False,
        "model_inference_calls":0,"optimizer_steps":0,
        "h140_sealed_blind_calls":0,"Hala35B_updated":False,
        "private_cases":rows,
    }
    (private/"manifest.json").write_text(
        json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    return {k:v for k,v in doc.items() if k!="private_cases"}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--owner-root",required=True)
    a=p.parse_args()
    print(json.dumps(collect(a.source,a.owner_root),indent=2,sort_keys=True))

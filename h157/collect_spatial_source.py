"""H157 frozen original GUI source for spatially varied native actions.

Creates fresh TRAIN280 and DEV72 from original MIT MiniWoB++.
Source oracle reads visible target labels only for supervision; execution
uses the same DOM-free native mouse/keyboard controller as H156.
NEVER uses H156 consumed DEV as train or a new evaluation cohort.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream,center
from h156.action_contract_v2 import action_from_dict_v2,native_mouse_keyboard
from h157.action_contract_v3 import SEQUENCES

SCHEMA="hala.h157.original-spatial-action-train-dev-v1"
APP_SHA="7fd85d71a4b60325c6585396ec4f48377d049838"
COUNTS={
    "TRAIN":{"click-test":64,"click-button":64,
             "enter-text":64,"enter-text-dynamic":64,"choose-list":24},
    "DEV":{"click-test":16,"click-button":16,
           "enter-text":16,"enter-text-dynamic":16,"choose-list":8},
}
EXPECTED_TRAIN=280
EXPECTED_DEV=72
SEED_BASE=550000


def prior_media_shas(owner):
    owner=Path(owner)
    folders=(
        owner/"private-h154"/"source-only-miniwob-v1"/"screenshots",
        owner/"private-h154"/"source-only-miniwob-v2"/"screenshots",
        owner/"private-h155"/"source-train-dev-v1"/"images",
        owner/"private-h156"/"miniwob-action-train-dev-v2"/"images",
        owner/"private-h157"/"smoke-new-tasks",
    )
    seen=set()
    for folder in folders:
        if not folder.is_dir():
            raise ValueError("H157 cannot omit historic original source exclusions")
        for path in folder.glob("*.png"):
            seen.add(hashlib.sha256(path.read_bytes()).hexdigest())
    if len(seen)!=146:
        raise ValueError("H157 expected all 146 prior original/smoke image SHA exclusions")
    return seen


def owner_reference_actions(page,task):
    if task=="click-test":
        actions=[{"kind":"click","point":list(
            center(page.locator("#subbtn")))}]
    elif task=="click-button":
        query=page.locator("#query").inner_text(timeout=3000).strip()
        match=re.fullmatch(r'Click on the "(.+)" button[.]',query)
        if not match:
            raise ValueError("Original multi-button visible goal no longer supported")
        options=page.locator("#area button").all()
        matching=[button for button in options
                  if button.inner_text(timeout=3000).strip()==match.group(1)]
        if not matching:
            raise ValueError("Original button goal has no visible matching element")
        actions=[{"kind":"click","point":list(center(matching[0]))}]
    elif task in ("enter-text","enter-text-dynamic"):
        visible=page.locator("#query .bold").inner_text(timeout=3000).strip()
        if not visible or len(visible)>128:
            raise ValueError("Original visible text instruction malformed")
        actions=[
            {"kind":"click","point":list(center(page.locator("#tt")))},
            {"kind":"type","text":visible},
            {"kind":"submit","point":list(center(page.locator("#subbtn")))}
        ]
    elif task=="choose-list":
        instruction=page.locator("#query").inner_text(timeout=3000).strip()
        match=re.fullmatch(r"Select (.+) from the list and click Submit[.]",
                           instruction)
        if not match:
            raise ValueError("Original dropdown visible target missing")
        actions=[
            {"kind":"select","point":list(center(page.locator("#options"))),
             "text":match.group(1)},
            {"kind":"submit","point":list(center(page.locator("#area button").first))}
        ]
    else:
        raise ValueError("H157 original task is unregistered")
    checked=tuple(action_from_dict_v2(a).kind for a in actions)
    if checked!=SEQUENCES[task]:
        raise ValueError("H157 oracle labels violate typed native action contract")
    return {"actions":actions}


def collect_episode(browser,html_root,task,split,index,seed,image_dir):
    context=browser.new_context(
        viewport={"width":960,"height":720},device_scale_factor=1,
        locale="en-US",service_workers="block")
    try:
        page=context.new_page()
        page.goto((html_root/(task+".html")).resolve().as_uri(),
                  wait_until="load",timeout=20000)
        page.evaluate("(seed)=>Math.seedrandom(String(seed))",seed)
        page.locator("#sync-task-cover").click(timeout=5000)
        page.wait_for_function(
            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
            timeout=5000)
        instruction=page.locator("#query").inner_text(timeout=3000).strip()
        if not instruction:
            raise ValueError("H157 original UI instruction absent")
        screenshot=page.screenshot(type="png")
        label=owner_reference_actions(page,task)
        outcome=native_mouse_keyboard(
            page,tuple(action_from_dict_v2(a) for a in label["actions"]))
        page.wait_for_function("() => WOB_DONE_GLOBAL === true",timeout=5000)
        state=page.evaluate(
            "()=>({done:WOB_DONE_GLOBAL,reward:WOB_RAW_REWARD_GLOBAL,"
            "episodes:WOB_EPISODE_ID})")
        if (outcome["status"]!="ACTIONS_SENT_NATIVE" or
            not state["done"] or state["reward"]<=0 or
            state["episodes"]!=1):
            raise ValueError("H157 original native reference failed")
        image=f"{split.lower()}-{task}-{index:03d}.png"
        (image_dir/image).write_bytes(screenshot)
        return {
            "split":split,"task":task,"seed":seed,
            "language":"en",
            "action_sequence":list(SEQUENCES[task]),
            "image":image,
            "image_sha256":hashlib.sha256(screenshot).hexdigest(),
            "instruction":instruction,
            "training_supervision":label,
            "original_positive_native_reward":True,
            "controller_uses_only_mouse_keyboard":True,
            "model_predictions_used":False,
        }
    finally:
        context.close()


def run(source,owner):
    from playwright.sync_api import sync_playwright
    owner=Path(owner)
    html_root=admit_upstream(source)
    root=owner/"private-h157"/"source-spatial-train-dev-v1"
    if root.exists():
        raise FileExistsError("H157 source create-only; do not repeat or overwrite")
    excluded=prior_media_shas(owner)
    root.mkdir(parents=True,exist_ok=False)
    (root/"SOURCE_ONE_SHOT.lock").write_text(SCHEMA+"\n",encoding="utf8")
    images=root/"images"
    images.mkdir()
    cases=[]
    with sync_playwright() as playwright:
        options={"headless":True}
        bin=_chromium_binary()
        if bin:
            options["executable_path"]=bin
        browser=playwright.chromium.launch(**options,timeout=20000)
        try:
            for split,quotas in COUNTS.items():
                for task_index,(task,count) in enumerate(quotas.items()):
                    for i in range(count):
                        seed=(SEED_BASE+int(split=="DEV")*100000+
                              task_index*10000+i)
                        row=collect_episode(
                            browser,html_root,task,split,i,seed,images)
                        if row["image_sha256"] in excluded:
                            raise ValueError("H157 screenshot reuses an earlier original")
                        excluded.add(row["image_sha256"])
                        cases.append(row)
                        if len(cases)%32==0:
                            print("H157_REAL_ORIGINAL_NATIVE_SOURCE",
                                  len(cases),EXPECTED_TRAIN+EXPECTED_DEV,
                                  flush=True)
        finally:
            browser.close()
    observed=Counter((r["split"],r["task"]) for r in cases)
    expected={(split,task):count for split,quotas in COUNTS.items()
              for task,count in quotas.items()}
    if (observed!=expected or len(cases)!=EXPECTED_TRAIN+EXPECTED_DEV or
        len(excluded)!=146+len(cases)):
        raise ValueError("H157 genuine original train/dev split not qualified")
    report={
        "schema":SCHEMA,
        "status":"H157_FRESH_ORIGINAL_MULTI_TASK_SOURCE_READY",
        "original_upstream_sha":APP_SHA,
        "original_license":"MIT",
        "train":EXPECTED_TRAIN,
        "dev":EXPECTED_DEV,
        "prior_original_sha_exclusions":146,
        "new_unique_original_images":len(cases),
        "positive_native_source_reference_episodes":len(cases),
        "task_counts":COUNTS,
        "dev_is_blind":False,
        "neural_model_inferences":0,
        "optimizer_steps":0,
        "h140_sealed_blind_calls":0,
        "no_paid_compute":True,
        "hala_flagship_35B_unchanged":True,
        "private_cases":cases,
    }
    (root/"manifest.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
        encoding="utf8")
    return {k:v for k,v in report.items() if k!="private_cases"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--owner-root",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.source,args.owner_root),sort_keys=True,indent=2))

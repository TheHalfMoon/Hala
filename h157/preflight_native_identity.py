"""H157 one-time screenshot identity preflight BEFORE model-native DEV.

No model calls or native actions; checks the 72 frozen source screenshots
replay pixel-identically under the saved original seed and Chromium.
Do not claim this probe is a neural evaluation or independent blind score.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream
from h157.audit_spatial_source import audit as audit_source
from h157.collect_spatial_source import COUNTS

OUTPUT="H157_FROZEN_DEV_NATIVE_SCREENSHOT_IDENTITY_V1.json"


def run(owner,source):
    owner=Path(owner)
    base=admit_upstream(source)
    source_receipt=audit_source(owner)
    private=owner/"private-h157"
    dest=private/OUTPUT
    if dest.exists():
        raise FileExistsError("H157 owner screenshot identity preflight once only")
    manifest=json.loads((private/"source-spatial-train-dev-v1"/
        "manifest.json").read_text("utf8"))
    dev=[case for case in manifest["private_cases"] if case["split"]=="DEV"]
    if len(dev)!=72:
        raise ValueError("H157 frozen DEV cohort count must be 72")
    from playwright.sync_api import sync_playwright
    passed=Counter()
    with sync_playwright() as pw:
        options={"headless":True}
        exe=_chromium_binary()
        if exe:
            options["executable_path"]=exe
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            for row in dev:
                context=browser.new_context(
                    viewport={"width":960,"height":720},
                    device_scale_factor=1,locale="en-US",service_workers="block")
                try:
                    page=context.new_page()
                    page.goto((base/(row["task"]+".html")).resolve().as_uri(),
                              wait_until="load",timeout=20000)
                    page.evaluate("(seed)=>Math.seedrandom(String(seed))",
                                  row["seed"])
                    page.locator("#sync-task-cover").click(timeout=5000)
                    page.wait_for_function(
                        "() => document.querySelector('#sync-task-cover').style.display === 'none'",
                        timeout=5000)
                    raw=page.screenshot(type="png")
                    observed=hashlib.sha256(raw).hexdigest()
                    if (observed!=row["image_sha256"] or
                        page.locator("#query").inner_text(timeout=3000).strip()
                        !=row["instruction"]):
                        raise ValueError("H157 frozen DEV pixel/visible text identity failed")
                    passed[row["task"]]+=1
                finally:
                    context.close()
                if sum(passed.values())%24==0:
                    print("H157_ORIGINAL_DEV_PIXEL_IDENTITY",
                          sum(passed.values()),72,flush=True)
        finally:
            browser.close()
    if dict(passed)!=COUNTS["DEV"]:
        raise ValueError("H157 frozen DEV stratified replay incomplete")
    summary={
        "schema":"hala.h157.owner-original-dev-identity-check.v1",
        "status":"H157_ORIGINAL_72_DEV_SCREENSHOTS_PIXEL_IDENTICAL",
        "source_manifest_sha256":source_receipt["manifest_sha256"],
        "original_dev_screenshots":72,
        "original_tasks":dict(passed),
        "neural_inference_calls":0,
        "optimizer_steps":0,
        "actual_model_native_reward_trials":0,
        "h140_blind_calls":0,
        "paid_compute":False,
        "release_qualified":False,
    }
    dest.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",
                    encoding="utf8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    parser.add_argument("--source",required=True)
    options=parser.parse_args()
    print(json.dumps(run(options.owner_root,options.source),sort_keys=True,indent=2))

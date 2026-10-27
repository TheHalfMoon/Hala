"""H156 true native browser replay of frozen model DEV predictions (one-shot).

DEV diagnostic only, not an independent unseen-app BLIND result.
No DOM/ground truth passed into model action execution. Original reward
and saved input screenshot hash are read only for scoring/identity.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream
from h156.action_contract_v2 import parse_v2,native_mouse_keyboard
from h156.audit_train_dev import audit
from h156.train_gpu import ADAPTER_SHA

LOCK="H156_DEV_NEURAL_NATIVE_REPLAY_CONSUMED.lock"
RESULT="H156_DEV_ACTUAL_MODEL_BROWSER_REWARDS.json"


def run(owner,source,model_result):
    owner=Path(owner)
    app=admit_upstream(source)
    check=audit(owner)
    train=owner/"private-h156"/"miniwob-action-train-dev-v2"
    result_dir=Path(model_result)
    file=result_dir/"private-h156-train-receipt.json"
    report=json.loads(file.read_text("utf8"))
    if (report.get("status")!="H156_COMPLETED_REAL_TYPED_ACTION_OPTIMIZER_STEPS"
        or report.get("actual_optimizer_steps")!=24 or
        report.get("source_manifest_sha256")!=check["source_sha256"]
        or report.get("frozen_initial_adapter_sha256")!=ADAPTER_SHA
        or report.get("h140_sealed_blind_calls")!=0):
        raise ValueError("H156 genuine optimizer receipt/source not proven")
    checkpoint=report["updated_action_adapter_sha256"]
    if not isinstance(checkpoint,str) or len(checkpoint)!=64:
        raise ValueError("H156 learned model checkpoint SHA missing")
    cases=json.loads((train/"manifest.json").read_text("utf8"))["private_cases"]
    dev={r["image_sha256"]:r for r in cases if r["split"]=="DEV"}
    if len(dev)!=12:
        raise ValueError("H156 DEV cohort changed")
    predictions={}
    for role in ("before","after"):
        data=json.loads((result_dir/(
            "private-h156-dev-"+role+".json")).read_text("utf8"))
        if not isinstance(data,list) or len(data)!=12:
            raise ValueError("H156 model DEV predictions incomplete")
        by_sha={row["image_sha"]:row for row in data}
        if len(by_sha)!=12 or set(by_sha)!=set(dev):
            raise ValueError("H156 neural predictions dev SHA mismatch")
        if any(row["task"]!=dev[row["image_sha"]]["task"]
               for row in data):
            raise ValueError("H156 neural DEV task labels mismatched")
        predictions[role]=by_sha
    output=owner/"private-h156"/RESULT
    lock=owner/"private-h156"/LOCK
    if output.exists() or lock.exists():
        raise FileExistsError("H156 native model DEV cohort consumed")
    lock.write_text(json.dumps({
        "status":"H156_ONE_SHOT_DEV_USED_FOR_NEURAL_REPLAY",
        "source_manifest_sha256":check["source_sha256"],
        "model_checkpoint_sha256":checkpoint,
        "h140_sealed_blind_calls":0
    },sort_keys=True)+"\n",encoding="utf8")
    from playwright.sync_api import sync_playwright
    episodes=[]
    with sync_playwright() as playwright:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:
            options["executable_path"]=binary
        browser=playwright.chromium.launch(**options,timeout=20000)
        try:
            for case in dev.values():
                for role,model_sha in (("before",ADAPTER_SHA),
                                       ("after",checkpoint)):
                    raw=predictions[role][case["image_sha"]]["raw"]
                    context=browser.new_context(
                        viewport={"width":960,"height":720},
                        device_scale_factor=1,locale="en-US",
                        service_workers="block")
                    try:
                        page=context.new_page()
                        page.goto((app/(case["task"]+".html")).resolve().as_uri(),
                                  wait_until="load",timeout=20000)
                        page.evaluate("(seed)=>Math.seedrandom(String(seed))",
                                      case["seed"])
                        page.locator("#sync-task-cover").click(timeout=5000)
                        page.wait_for_function(
                            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
                            timeout=5000)
                        screenshot=page.screenshot(type="png")
                        if (hashlib.sha256(screenshot).hexdigest()
                            !=case["image_sha256"] or
                            page.locator("#query").inner_text().strip()
                            !=case["instruction"]):
                            raise ValueError("H156 native model input changed from frozen screenshot")
                        parse_valid=False
                        abstained=False
                        executed=0
                        reward=0
                        try:
                            actions=parse_v2(raw,task=case["task"],
                                             model_checkpoint_sha=model_sha)
                            parse_valid=True
                            if len(actions)==1 and actions[0].kind=="abstain":
                                abstained=True
                            else:
                                feedback=native_mouse_keyboard(page,actions)
                                executed=feedback["actions_executed"]
                                try:
                                    page.wait_for_function(
                                        "() => window.WOB_DONE_GLOBAL === true",
                                        timeout=1600)
                                except Exception:
                                    pass
                                state=page.evaluate(
                                    "() => ({done: WOB_DONE_GLOBAL,"
                                    " reward: WOB_RAW_REWARD_GLOBAL,"
                                    " episodes: WOB_EPISODE_ID})")
                                if state["done"] and state["episodes"]==1:
                                    reward=state["reward"]
                        except ValueError:
                            pass
                        episodes.append({
                            "role":role,"task":case["task"],
                            "image_sha":case["image_sha256"],
                            "parse_valid":parse_valid,
                            "abstained":abstained,
                            "model_actions_executed":executed,
                            "native_success":bool(reward>0),
                            "native_reward_positive":reward>0,
                        })
                    finally:
                        context.close()
                if len(episodes)%8==0:
                    print("H156_REAL_MODEL_NATIVE_DEV_PAIRED",
                          len(episodes),24,flush=True)
        finally:
            browser.close()
    counter=Counter()
    for r in episodes:
        counter[r["role"]+"_passed"]+=int(r["native_success"])
        counter[r["role"]+"_parse_valid"]+=int(r["parse_valid"])
        counter[r["role"]+"_abstentions"]+=int(r["abstained"])
    summary={
        "schema":"hala.h156.true-native-neural-dev-replay.v1",
        "status":"COMPLETE_MODEL_DRIVEN_ORIGINAL_MINIWOB_DEV_REPLAY",
        "dev_tasks":12,"real_model_native_browser_trials":24,
        "frozen_initial_lora_sha":ADAPTER_SHA,
        "new_lora_sha":checkpoint,
        "model_before_native_success":counter["before_passed"],
        "model_after_native_success":counter["after_passed"],
        "model_before_parse_valid":counter["before_parse_valid"],
        "model_after_parse_valid":counter["after_parse_valid"],
        "model_before_abstentions":counter["before_abstentions"],
        "model_after_abstentions":counter["after_abstentions"],
        "dev_not_blind":True,
        "cross_app_transfer_proven":False,
        "h140_blind_calls":0,"paid_gpu":False,
        "Hala_flagship_35B_updated":False,
        "model_published":False,"release_qualified":False,
    }
    output.write_text(json.dumps({
        "summary":summary,"owner_private_cases":episodes},
        indent=2,sort_keys=True)+"\n",encoding="utf8")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--result-dir",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner_root,args.source,args.result_dir),
                     indent=2,sort_keys=True))

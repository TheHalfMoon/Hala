"""H157 one-shot native Chromium paired DEV trial with actual GPU model outputs.

Allowed model input: screenshot plus visible instruction. Allowed controller
input: model JSON only. Original MiniWoB source DOM is used solely by the
harness BEFORE action for SHA/visible-instruction identity and AFTER action
for the original reward/episode count. Never use owner gold during actions.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h154.miniwob_source_probe import admit_upstream
from h156.action_contract_v2 import native_mouse_keyboard
from h157.action_contract_v3 import parse_v3
from h157.audit_spatial_source import audit as source_audit
from h157.audit_gpu_learning import audit as gpu_audit
from h157.train_gpu import START_ADAPTER_SHA

LOCK="H157_MODEL_NATIVE_DEV_ONCE_CONSUMED.lock"
OUTPUT="H157_MODEL_NATIVE_DEV_BROWSER_REWARD.json"
ROLES=("before","after")


def preflight(owner, app_source, result_root, private_archive):
    owner=Path(owner)
    original=admit_upstream(app_source)
    frozen=source_audit(owner)
    verified=gpu_audit(result_root, private_archive)
    if verified["source_manifest_sha256"]!=frozen["manifest_sha256"]:
        raise ValueError("H157 actual model checkpoint/source provenances differ")
    private=owner/"private-h157"
    src=json.loads((private/"source-spatial-train-dev-v1"/"manifest.json")
                   .read_text("utf8"))
    dev={r["image_sha256"]:r for r in src["private_cases"] if r["split"]=="DEV"}
    if len(dev)!=72:
        raise ValueError("H157 DEV 72 source images absent or duplicated")
    modelroot=Path(result_root)/"hala-h157-spatial-action-lora"
    predictions={}
    for role in ROLES:
        path=modelroot/("private-h157-dev-"+role+".json")
        rows=json.loads(path.read_text("utf8"))
        if not isinstance(rows,list) or len(rows)!=72:
            raise ValueError("H157 model outputs missing")
        lookup={}
        for row in rows:
            if (not isinstance(row,dict) or
                set(row)!={"task","image_sha","raw","valid","sequence","text_correct"} or
                not isinstance(row["raw"],str) or row["image_sha"] in lookup or
                row["image_sha"] not in dev or
                row["task"]!=dev[row["image_sha"]]["task"]):
                raise ValueError("H157 predictions not for frozen DEV images")
            lookup[row["image_sha"]]=row["raw"]
        if set(lookup)!=set(dev):
            raise ValueError("H157 incomplete model prediction DEV coverage")
        predictions[role]=lookup
    # All admission and parsing must happen BEFORE the one-shot lock.
    adapter_sha={"before":START_ADAPTER_SHA,
                 "after":verified["new_adapter_sha256"]}
    parsed={}
    for image_sha,case in dev.items():
        for role in ROLES:
            raw=predictions[role][image_sha]
            try:
                actions=parse_v3(raw,task=case["task"],
                                 model_checkpoint_sha=adapter_sha[role])
            except ValueError:
                actions=None
            parsed[(image_sha,role)]=actions
    return original,frozen,verified,dev,parsed


def run(owner,app_source,result_root,private_archive):
    owner=Path(owner)
    app,frozen,verified,dev,parsed=preflight(
        owner,app_source,result_root,private_archive)
    private=owner/"private-h157"
    lock=private/LOCK
    outfile=private/OUTPUT
    if lock.exists() or outfile.exists():
        raise FileExistsError("H157 DEV cohort already consumed; never rerun")
    # Write EXCLUSIVE once, so accidental parallel invocations cannot double-use.
    with lock.open("x",encoding="utf8") as stream:
        stream.write(json.dumps({
            "schema":"hala.h157.owner-neural-native-dev-once.v1",
            "status":"H157_DEV_NATIVE_ONESHOT_CONSUMED",
            "manifest_sha256":frozen["manifest_sha256"],
            "before_adapter_sha256":START_ADAPTER_SHA,
            "after_adapter_sha256":verified["new_adapter_sha256"],
            "private_dev_screenshots":72,
            "h140_blind_calls":0,
        },sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    episodes=[]
    with sync_playwright() as pw:
        options={"headless":True}
        exe=_chromium_binary()
        if exe:
            options["executable_path"]=exe
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            for case in dev.values():
                for role in ROLES:
                    ctx=browser.new_context(
                        viewport={"width":960,"height":720},
                        device_scale_factor=1,locale="en-US",
                        service_workers="block")
                    try:
                        page=ctx.new_page()
                        page.goto((app/(case["task"]+".html")).resolve().as_uri(),
                                  wait_until="load",timeout=20000)
                        page.evaluate("(seed)=>Math.seedrandom(String(seed))",
                                      case["seed"])
                        page.locator("#sync-task-cover").click(timeout=5000)
                        page.wait_for_function(
                            "() => document.querySelector('#sync-task-cover').style.display === 'none'",
                            timeout=5000)
                        picture=page.screenshot(type="png")
                        if (hashlib.sha256(picture).hexdigest()!=case["image_sha256"] or
                            page.locator("#query").inner_text(timeout=3000).strip()
                            !=case["instruction"]):
                            raise ValueError("H157 task screenshot/instruction replay diverged")
                        acts=parsed[(case["image_sha256"],role)]
                        accepted=acts is not None
                        abstained=bool(acts is not None and
                                       len(acts)==1 and acts[0].kind=="abstain")
                        sent=0
                        native_success=False
                        if accepted and not abstained:
                            result=native_mouse_keyboard(page,acts)
                            if result["status"]!="ACTIONS_SENT_NATIVE":
                                raise ValueError("Model actions not sent to native browser")
                            sent=result["actions_executed"]
                            try:
                                page.wait_for_function(
                                    "() => WOB_DONE_GLOBAL === true",timeout=1350)
                            except Exception:
                                pass
                            reward=page.evaluate(
                                "() => ({done: WOB_DONE_GLOBAL,"
                                " raw: WOB_RAW_REWARD_GLOBAL,"
                                " episodes: WOB_EPISODE_ID})")
                            native_success=bool(reward["done"] and
                                reward["episodes"]==1 and reward["raw"]>0)
                        episodes.append({
                            "role":role,"task":case["task"],
                            "image_sha":case["image_sha256"],
                            "parse_valid":accepted,"abstained":abstained,
                            "model_actions_executed":sent,
                            "native_success":native_success,
                            "native_reward_positive":native_success,
                        })
                    finally:
                        ctx.close()
                if len(episodes)%24==0:
                    print("H157_REAL_NEURAL_CHROMIUM_DEV",len(episodes),
                          144,flush=True)
        finally:
            browser.close()
    results=Counter()
    per_task=Counter()
    for row in episodes:
        results[row["role"]+"_successful"]+=int(row["native_success"])
        results[row["role"]+"_accepted"]+=int(row["parse_valid"])
        results[row["role"]+"_abstain"]+=int(row["abstained"])
        per_task[(row["role"],row["task"])]+=int(row["native_success"])
    report={
        "schema":"hala.h157.private-real-neural-browser-dev-paired.v1",
        "status":"H157_COMPLETE_REAL_MODEL_144_NATIVE_BROWSER_DEV_EPISODES",
        "dev_source_manifest_sha256":frozen["manifest_sha256"],
        "baseline_h156_adapter_sha256":START_ADAPTER_SHA,
        "trained_h157_adapter_sha256":verified["new_adapter_sha256"],
        "original_chromium_model_action_episodes":144,
        "unique_dev_screenshots":72,
        "before_native_success":results["before_successful"],
        "after_native_success":results["after_successful"],
        "before_valid_model_action_json":results["before_accepted"],
        "after_valid_model_action_json":results["after_accepted"],
        "before_model_abstentions":results["before_abstain"],
        "after_model_abstentions":results["after_abstain"],
        "by_task":{task:{arm:per_task[(arm,task)] for arm in ROLES}
                   for task in ("click-test","click-button","enter-text",
                                "enter-text-dynamic","choose-list")},
        "same_app_dev_not_blind":True,"multiapp_transfer_proven":False,
        "model_weights_published":False,"Hala35B_unchanged":True,
        "paid_gpu":False,"h140_blind_calls":0,
    }
    outfile.write_text(json.dumps({
        "summary":report,"owner_private_cases":episodes},
        indent=2,sort_keys=True)+"\n",encoding="utf8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("--result-root",required=True)
    parser.add_argument("--source-archive",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.owner_root,args.source,args.result_root,
                         args.source_archive),sort_keys=True,indent=2))

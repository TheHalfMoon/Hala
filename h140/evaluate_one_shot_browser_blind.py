"""Owner-only ONE-SHOT sealed H140 model-driven real browser drag benchmark.

The model sees only native screenshots + Arabic/English instructions.
Unmodified original MIT browser scenes are SHA-identically replayed BEFORE
every predicted drag. The reference oracle is read only on owner Windows,
AFTER pretrained+LoRA selection, never included in Kaggle inference.
This evaluator is fail-closed and one-shot: no iterative BLIND tuning.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import mouse_drag,serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.replay_blind_source import (
    CANDIDATES,source_pairing,setup_context,position_original_scene
)
from h140.blind_original_capture import board_for,param,moved
from h140.blind_source_preflight import SOURCES,check_original_source
from h140.export_blind_redacted import OWNER_BLIND_SHA
from h140.evaluate_v3_gate import evaluate,actual_adapter_hash
from h140.qualification import digest

SCHEMA="hala.h140.private-once-original-two-app-model-predictions.v1"
INPUT_SHA="ecde02bd4658a6d7a4302e522064d5ec42066837ee30dc7d93a5dd9aa90b7ef9"
POINT_KEYS={"id","app","language","start_point","end_point"}
RECEIPT_NAME="ONE_SHOT_EVALUATION_ALREADY_STARTED.json"

def valid_point(p):
    if p is None:
        return False
    return (isinstance(p,list) and len(p)==2
            and all(type(v) in (int,float) and math.isfinite(v) for v in p)
            and 0<=p[0]<WIDTH and 0<=p[1]<HEIGHT)

def load_predictions(path,chosen_sha,owner_root):
    data=json.loads(Path(path).read_text("utf8"))
    if (data.get("schema")!=SCHEMA or
        data.get("owner_original_manifest_sha256")!=OWNER_BLIND_SHA or
        data.get("redacted_manifest_sha256")!=INPUT_SHA or
        data.get("adapter_sha256")!=chosen_sha or
        data.get("model_evaluation_role")!="one_shot_blind_inference_only" or
        data.get("trained_on_blind") is not False or
        data.get("model_calls")!=64):
        raise ValueError("H140 selected original unseen model predictions unsupported")
    pairs=source_pairing(owner_root)
    allowed={
        (frozen["id"],app,frozen["language"])
        for app,items in pairs.items() for original,frozen in items
    }
    rows=data.get("predictions")
    if not isinstance(rows,list) or len(rows)!=32:
        raise ValueError("H140 exact blind32 model task count required")
    found={}
    for r in rows:
        if not isinstance(r,dict) or set(r)!=POINT_KEYS:
            raise ValueError("H140 model predictions must not embed any oracle keys")
        identity=(r["id"],r["app"],r["language"])
        if identity not in allowed or identity in found:
            raise ValueError("H140 unseen original app task identity mismatch")
        if any(p is not None and not valid_point(p)
               for p in (r["start_point"],r["end_point"])):
            raise ValueError("H140 predicted pixels must be numeric and on-screen")
        found[identity]=r
    if set(found)!=allowed:
        raise ValueError("H140 all original blind task identities required")
    return found

def score_original_blind(owner_root,predictions,v2_dir,v3_dir):
    root=Path(owner_root)
    v2_dir,v3_dir=Path(v2_dir),Path(v3_dir)
    gate=evaluate(v2_dir,v3_dir)
    if gate["status"]!="ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND":
        raise ValueError("H140 V3 still failed DEV; ABSOLUTELY DO NOT inspect BLIND labels")
    v3_receipt=json.loads((v3_dir/"private-h140-v3-learning-evidence.json").read_text("utf8"))
    chosen_sha=actual_adapter_hash(v3_dir,v3_receipt)
    expected=load_predictions(predictions,chosen_sha,root)
    # Import the optional native browser dependency ONLY after the DEV
    # authorization and no-gold prediction schema pass. The CPU-only CI
    # must reject a NO-GO without installing Playwright or launching Chrome.
    from playwright.sync_api import sync_playwright

    marker=root/"frozen-source-v1"/"blind_test"/RECEIPT_NAME
    if marker.exists():
        raise ValueError("H140 one-shot original blind already started; no silent retest")
    locked=json.dumps({
        "schema":"hala.h140.one-shot-source-test-lock.v1",
        "model_adapter_sha256":chosen_sha,
        "redacted_manifest_sha256":INPUT_SHA,
        "owner_gold_shared_with_model":False,
        "status":"BLIND_EVALUATION_CONSUMED_NO_RETRY",
    },sort_keys=True,indent=2)+"\n"
    # Exclusive create intentionally consumes BLIND, even on mid-test failures.
    with marker.open("x",encoding="utf8") as f:
        f.write(locked)

    by_app=source_pairing(root)
    outcomes=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            for app,items in by_app.items():
                check_original_source(root/"upstream"/app,SOURCES[app])
                with serve(root/"upstream"/app) as url:
                    for original,frozen in items:
                        attempt=original["attempt_index"]
                        n,source_index,_,_,_=param(attempt)
                        seed=board_for(app,attempt,n,source_index)
                        key=(frozen["id"],app,frozen["language"])
                        pred=expected[key]
                        ctx=setup_context(browser,seed,app)
                        try:
                            page=ctx.new_page()
                            page.goto(url,wait_until="commit",timeout=18000)
                            position_original_scene(page,app,attempt,seed)
                            native=page.screenshot(type="png")
                            if digest(native)!=frozen["image_sha256"]:
                                raise ValueError("H140 model-directed scene NOT identical to frozen original")
                            start,end=pred["start_point"],pred["end_point"]
                            source_box,goal_box=frozen["start_bbox"],frozen["end_bbox"]
                            in_box=lambda p,b: bool(valid_point(p) and
                                b["x"]<=p[0]<=b["x"]+b["width"] and
                                b["y"]<=p[1]<=b["y"]+b["height"])
                            start_hit=in_box(start,source_box)
                            end_hit=in_box(end,goal_box)
                            changed=False
                            persisted=False
                            if valid_point(start) and valid_point(end):
                                mouse_drag(page,start,end)
                                page.wait_for_timeout(140)
                                current=json.loads(page.evaluate(
                                    "(k)=>localStorage.getItem(k)",seed["storage"]))
                                changed=moved(app,seed,current)
                                page.reload(wait_until="commit",timeout=18000)
                                last=json.loads(page.evaluate(
                                    "(k)=>localStorage.getItem(k)",seed["storage"]))
                                persisted=moved(app,seed,last)
                            outcomes.append({
                                "id":frozen["id"],"app":app,"language":frozen["language"],
                                "start_point_in_original_card":bool(start_hit),
                                "end_point_in_original_target":bool(end_hit),
                                "both_model_points_in_targets":bool(start_hit and end_hit),
                                "real_native_drag_changed_task":bool(changed),
                                "real_native_drag_survived_reload":bool(changed and persisted),
                                "source_image_sha_match":True,
                            })
                        finally:ctx.close()
        finally:browser.close()
    if len(outcomes)!=32:
        raise ValueError("H140 all original one-shot browser outcomes required")
    counts=defaultdict(Counter)
    for entry in outcomes:
        for key in ("__all__",entry["app"],entry["app"]+"/"+entry["language"]):
            item=counts[key]
            item["cases"]+=1
            for field in ("start_point_in_original_card",
                          "end_point_in_original_target",
                          "both_model_points_in_targets",
                          "real_native_drag_changed_task",
                          "real_native_drag_survived_reload"):
                item[field]+=int(entry[field])
    result={
        "schema":"hala.h140.one-shot-real-persisted-browser-transfer.v1",
        "status":"ACTUAL_BLIND_ONE_SHOT_COMPLETE_NO_AUTOMATIC_RELEASE",
        "model_checkpoint_sha256":chosen_sha,
        "owner_original_manifest_sha256":OWNER_BLIND_SHA,
        "predictions_blind_train_leak":False,
        "cases":32,"unique_original_app_codebases":2,
        "per_app_language_counts":{a:dict(b) for a,b in counts.items()},
        "flagship_Hala_35B_updated":False,"paid_compute_used":False,
        "release_qualified":False,
    }
    result_path=marker.parent/"ONE_SHOT_PRIVATE_BROWSER_OUTCOMES.json"
    with result_path.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":result,"private_case_outcomes":outcomes},
                           indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--v2-evidence",required=True)
    p.add_argument("--v3-evidence",required=True)
    a=p.parse_args()
    print(json.dumps(score_original_blind(a.private_root,a.predictions,
                                          a.v2_evidence,a.v3_evidence),
                     indent=2,sort_keys=True))

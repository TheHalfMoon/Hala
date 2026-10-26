"""Owner-only exact-browser replay of sealed original H140 blind screenshot.

Source-only QA is NOT neural evaluation and never reveals target boxes to a
model. It guarantees a future model-driven drag may be tested on EXACTLY
the original state depicted to the model, not an accidentally easier scene.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.blind_original_capture import board_for,param,native_card,center
from h140.blind_source_preflight import SOURCES,check_original_source,only_local_requests
from h140.export_blind_redacted import OWNER_BLIND_SHA
from h140.qualification import load_split,digest

CANDIDATES={
    "ood-vanilla-complete":"blind-candidate-aisha16-v1",
    "ood-second-drag":"blind-candidate-michael16-v4",
}

def source_pairing(root):
    root=Path(root)
    seal=root/"frozen-source-v1"/"blind_test"
    payload=(seal/"manifest.json").read_bytes()
    if digest(payload)!=OWNER_BLIND_SHA:
        raise ValueError("H140 real-app source replay must use exact owner-blind seal")
    doc=load_split(seal,"blind_test")
    by_app={}
    for name,candidate in CANDIDATES.items():
        folder=root/candidate
        source=json.loads((folder/"manifest.json").read_text("utf8"))
        if (source.get("app")!=name
            or source.get("app_git_sha")!=SOURCES[name]["sha"]
            or source.get("status")!="PRIVATE_ORIGINAL_BLIND_SOURCE_GEOMETRY_READY"
            or source.get("training_eligible") is not False
            or source.get("blind_model_evaluated") is not False
            or source.get("model_neural_calls")!=0):
            raise ValueError("H140 selected original unseen app source provenance invalid")
        original=source.get("cases",[])
        frozen=[r for r in doc["cases"] if r["app"]==name]
        if len(original)!=16 or len(frozen)!=16:
            raise ValueError("H140 require 16 genuine task frames from EACH original blind app")
        for a,b in zip(original,frozen):
            if (a["image_sha256"]!=b["image_sha256"] or
                a["language"]!=b["language"] or
                a["app_git_sha"]!=b["app_git_sha"] or
                not isinstance(a.get("attempt_index"),int)):
                raise ValueError("H140 original screenshot lost source/replay identity")
        by_app[name]=list(zip(original,frozen))
    return by_app

def setup_context(browser,seed,app):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    key=seed["storage"]
    values=json.dumps(json.dumps(seed["seed"]))
    script=("if(localStorage.getItem("+json.dumps(key)+")===null){"+
            "localStorage.setItem("+json.dumps(key)+","+values+");")
    if app=="ood-second-drag":
        script+=("localStorage.setItem('currentBoardId',"+
                 json.dumps(seed["board_id"])+");")
    ctx.add_init_script(script+"}")
    ctx.route("**/*",only_local_requests)
    return ctx

def position_original_scene(page,app,attempt,seed):
    n,source_i,dest_i,_,scroll=param(attempt)
    if app=="ood-second-drag":
        dest_i=source_i
    src=native_card(page,app,"start",source_i,seed["id"])
    dst=native_card(page,app,"end",dest_i,seed["id"])
    src.wait_for(timeout=18000)
    dst.wait_for(timeout=18000)
    if app=="ood-second-drag":
        src.scroll_into_view_if_needed(timeout=15000)
        dst.scroll_into_view_if_needed(timeout=15000)
        # Frozen native source-v4 collector exact independent pane scroll.
        delta=(attempt*83)%165
        page.evaluate("(delta)=>{for(const column of document.querySelectorAll('.tasks-container[data-status=todo],.tasks-container[data-status=inprogress]')){column.scrollTop+=delta}}",delta)
        page.evaluate("(target)=>window.scrollTo(0,target)",(attempt*61)%148)
    else:
        pre_start=src.bounding_box()
        pre_end=dst.bounding_box()
        if not pre_start or not pre_end:
            raise ValueError("H140 source screenshot not reconstructable")
        last=max(center(pre_start)[1],center(pre_end)[1])
        page.evaluate("(y)=>window.scrollTo(0,y)",max(0,last-620)+scroll)
    page.wait_for_timeout(120)
    return src,dst

def source_only_replay(private_root):
    from playwright.sync_api import sync_playwright
    private_root=Path(private_root)
    pairs=source_pairing(private_root)
    total=Counter()
    with sync_playwright() as pw:
        launch={"headless":True}
        exe=_chromium_binary()
        if exe:launch["executable_path"]=exe
        browser=pw.chromium.launch(**launch,timeout=20000)
        try:
            for app,items in pairs.items():
                check_original_source(private_root/"upstream"/app,SOURCES[app])
                with serve(private_root/"upstream"/app) as url:
                    for original,frozen in items:
                        attempt=original["attempt_index"]
                        n,si,_,_,_=param(attempt)
                        seed=board_for(app,attempt,n,si)
                        ctx=setup_context(browser,seed,app)
                        try:
                            page=ctx.new_page()
                            page.goto(url,wait_until="commit",timeout=18000)
                            position_original_scene(page,app,attempt,seed)
                            photo=page.screenshot(type="png")
                            if digest(photo)!=frozen["image_sha256"]:
                                raise ValueError("H140 sealed source image is not exactly reproducible: "+
                                                 app+"/"+frozen["id"])
                            total[app]+=1
                        finally:
                            ctx.close()
                        if total[app]%4==0:
                            print("H140_OWNER_EXACT_NATIVE_BLIND_IMAGE_SOURCE_QA",
                                  app,total[app],flush=True)
        finally:
            browser.close()
    if dict(total)!={app:16 for app in SOURCES}:
        raise ValueError("H140 exact original source replay lacked all 32 blind tasks")
    return {"schema":"hala.h140.owner-blind-browser-original-exact-replay-qa.v1",
            "status":"EXACT_PIXEL_REPLAY_PASS_NO_NEURAL_CALLS",
            "cases":32,
            "original_source_apps":2,
            "exact_png_sha_matches":32,
            "model_predictions_accessed":0,
            "gold_shared_with_model":False,
            "training_on_blind":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--private-root",required=True)
    args=parser.parse_args()
    print(json.dumps(source_only_replay(args.private_root),indent=2,sort_keys=True))

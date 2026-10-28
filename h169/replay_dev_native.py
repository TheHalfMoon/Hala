"""H169 owner-Windows paired real Chromium DEV reward, one frozen pass.

Model sees ONLY the PNG and task. This native reward never provides reference
points/gold to models and does not use click-distance scoring. Once consumed,
do not use these DEV cases to tune/select hyperparameters or make further runs.
"""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import center,mouse_drag,serve
from h159.stable_source_qa import stable_png
from h164.native_action_contract import parse_action
from h169.infer_gpu import DEV_ZIP_SHA,VISION_WEIGHTS_SHA
from h169.source_dev import SCHEMA,cases,admitted_history
from h168.source import verify_sources

LOCK="H169_NEW_ORIGINAL_DEV_ONE_SHOT_CONSUMED.lock"
REWARD="H169_NATIVE_PAIRED_NEW_DEV_REWARD.json"

def sha(b):
    return hashlib.sha256(b).hexdigest()

def preflight(owner,upstream,train_source,source,predictions):
    owner=Path(owner);source=Path(source)
    verify_sources(upstream)
    prior=admitted_history(owner,train_source)
    doc=json.loads((source/"manifest.json").read_text("utf8"))
    rows=doc.get("private_cases")
    expected=cases()
    if (doc.get("schema")!=SCHEMA or doc.get("count")!=32
        or doc.get("split")!="DEV" or doc.get("reference_gold_uploaded") is not False
        or not isinstance(rows,list) or len(rows)!=32):
        raise ValueError("H169 original preregistered DEV manifest mismatched")
    for row,case in zip(rows,expected):
        if (any(row.get(k)!=v for k,v in case.items())
            or row.get("image_sha256") in prior
            or sha((source/row["image"]).read_bytes())!=row["image_sha256"]):
            raise ValueError("H169 original DEV/previous TRAIN image admission failed")
    doc=json.loads(Path(predictions).read_text("utf8"))
    if (doc.get("schema")!="hala.h169.frozen-paired-original-dev-predictions.v1"
        or doc.get("status")!="H169_PRIVATE_64_REAL_GPU_DEV_PREDICTIONS_COMPLETE"
        or doc.get("source_dev_zip_sha256")!=DEV_ZIP_SHA
        or doc.get("trained_vision_sha256")!=VISION_WEIGHTS_SHA
        or doc.get("real_gpu_calls")!=64 or doc.get("paired_dev_cases")!=32
        or doc.get("optimizer_steps")!=0
        or doc.get("owner_gold_on_gpu") is not False
        or doc.get("previous_consumed_ood_used") is not False
        or doc.get("paid_gpu") is not False):
        raise ValueError("H169 model frozen paired role provenance not qualified")
    vals=doc.get("predictions")
    if not isinstance(vals,list) or len(vals)!=64:
        raise ValueError("H169 frozen paired proposals missing")
    paired={}
    for row in vals:
        role=row.get("role");key=(row.get("case_index"),role)
        original=next((c for c in rows if c["index"]==row.get("case_index")),None)
        if (role not in ("before","after") or original is None or key in paired
            or row.get("image_sha256")!=original["image_sha256"]
            or row.get("app")!=original["app"]
            or row.get("language")!=original["language"]
            or row.get("action_type")!=original["action_type"]
            or row.get("parsed")!=parse_action(row.get("raw"))):
            raise ValueError("H169 modified, duplicate or wrong-role GPU proposal")
        paired[key]=row
    if set(paired)!={(c["index"],name) for c in rows
                      for name in ("before","after")}:
        raise ValueError("H169 all 64 paired DEV model outputs required")
    return rows,paired

def native(browser,url,row,model_row):
    ctx=browser.new_context(viewport={"width":960,"height":720},
            device_scale_factor=1,locale="en-US",service_workers="block")
    try:
        page=ctx.new_page()
        app=row["app"];n=row["variant"]
        if app=="choices":
            path="choices/public/test/text/index.html"
        else:
            path="nouislider/tests/origin.html"
        response=page.goto(url+path,wait_until="load",timeout=25000)
        if response is None or response.status!=200:
            raise ValueError("H169 actual original-app HTTP/source unavailable")
        if app=="choices":
            hook=("basic","edit-items","remove-button","unique-values")[n%4]
            host=page.locator('[data-test-hook="'+hook+'"]')
            widget=host.locator(".choices__input--cloned").first
            widget.wait_for(timeout=10000)
            host.scroll_into_view_if_needed()
            seed="halaseed"+str(168000+n)
            widget.click();widget.fill(seed);widget.press("Enter")
            if host.locator(".choices__item[data-item]").filter(
                    has_text=seed).count()!=1:
                raise ValueError("H169 original DEV seed invalid before model action")
            expected_text="halanew"+str(168000+n)
        else:
            page.locator("#slider1 .noUi-handle").wait_for(timeout=10000)
            page.evaluate(
                "([a,b]) => { slider1.noUiSlider.set(a); slider2.noUiSlider.set(b); }",
                [2+(n%5),2+(n//5)])
            selector="#slider1" if n%2==0 else "#slider2"
            old=float(page.evaluate(
                "(s)=>document.querySelector(s).noUiSlider.get()",selector))
        if sha(stable_png(page))!=row["image_sha256"]:
            raise ValueError("H169 frozen original DEV screenshot replay mismatch")
        proposal=model_row["parsed"]
        valid=proposal["valid"] is True and proposal["kind"]==row["action_type"]
        if valid and app=="choices":
            click=proposal["points"][0]
            page.mouse.click(*click)
            page.keyboard.type(proposal.get("text",""))
            page.keyboard.press("Enter")
            page.wait_for_timeout(70)
            success=host.locator(".choices__item[data-item]").filter(
                has_text=expected_text).count()==1
        elif valid and app=="nouislider":
            pts=proposal["points"]
            mouse_drag(page,pts[0],pts[1])
            new=float(page.evaluate(
                "(s)=>document.querySelector(s).noUiSlider.get()",selector))
            success=(math.isfinite(new) and new>old+0.5)
        else:
            success=False
        return {"success":bool(success),"valid_kind":bool(valid),
                "parsed_kind":proposal.get("kind")}
    finally:
        ctx.close()

def run(owner,upstream,train_source,source,predictions,output):
    from playwright.sync_api import sync_playwright
    owner=Path(owner);output=Path(output)
    if output.exists() or (owner/"private-h169"/LOCK).exists():
        raise ValueError("H169 frozen DEV result already exists; no repeated selection")
    rows,models=preflight(owner,upstream,train_source,source,predictions)
    results=[];summary=Counter()
    with serve(upstream) as url:
        with sync_playwright() as playwright:
            opts={"headless":True}
            exe=_chromium_binary()
            if exe:opts["executable_path"]=exe
            browser=playwright.chromium.launch(**opts)
            try:
                for case in rows:
                    for role in ("before","after"):
                        observed=native(browser,url,case,models[(case["index"],role)])
                        summary[role+"/calls"]+=1
                        summary[role+"/success"]+=int(observed["success"])
                        summary[role+"/"+case["language"]+"/success"]+=int(observed["success"])
                        summary[role+"/"+case["app"]+"/success"]+=int(observed["success"])
                        results.append(dict(index=case["index"],role=role,
                            app=case["app"],language=case["language"],
                            image_sha256=case["image_sha256"],**observed))
                        print("H169_NATIVE_DEV_REWARD",len(results),64,role,
                              observed["success"],flush=True)
            finally:browser.close()
    if len(results)!=64 or summary["before/calls"]!=32 or summary["after/calls"]!=32:
        raise ValueError("H169 original paired DEV replay incomplete")
    lookup={(r["index"],r["role"]):r["success"] for r in results}
    wins=sum(lookup[(c["index"],"after")] and not lookup[(c["index"],"before")]
             for c in rows)
    losses=sum(lookup[(c["index"],"before")] and not lookup[(c["index"],"after")]
               for c in rows)
    doc=dict(schema="hala.h169.new-32-original-dev-owner-browser-native-reward.v1",
        status="H169_FRESH_ORIGINAL_DEV_NATIVE_PAIRED_ONE_SHOT_CONSUMED",
        before_success=summary["before/success"],
        after_success=summary["after/success"],total=32,
        paired_wins=wins,paired_regressions=losses,counts=dict(summary),
        source_dev_zip_sha256=DEV_ZIP_SHA,trained_vision_sha256=VISION_WEIGHTS_SHA,
        evaluated_once=True,old_consumed_ood_used=False,
        model_weights_changed=False,private_owner_gold_on_gpu=False,
        reward_uses_native_state_not_gold_proximity=True,
        real_gpu_proposals=64,paid_gpu=False,flagship_35b_unchanged=True,
        private_outcomes=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n","utf8")
    lock=owner/"private-h169"/LOCK
    lock.parent.mkdir(exist_ok=True,parents=True)
    lock.write_text(json.dumps(dict(source_sha=DEV_ZIP_SHA,
               status="H169_NATIVE_DEV_CONSUMED_NO_REUSE"),sort_keys=True)+"\n","utf8")
    return {k:v for k,v in doc.items() if k not in ("private_outcomes",)}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("owner","upstream","train-source","source","predictions","output"):
        parser.add_argument("--"+key,required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.owner,a.upstream,a.train_source,a.source,
       a.predictions,a.output),indent=2,sort_keys=True))

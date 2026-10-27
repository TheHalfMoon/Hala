"""H164 owner-Windows ONE-SHOT native browser OOD reward on two new apps.

Only frozen neural model-generated pixel actions enter the click controller;
owner gold mouse points are neither supplied nor used to score by proximity.
"""
import argparse,json,math
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
from h130.browser_env import _chromium_binary
from h138.probe import serve
from h159.stable_source_qa import stable_png
from h164.new_ood_64 import SCHEMA,PRIVATE,PINS,TAGS,sources,cases
from h164.expand_288_train import digest
from h164.native_action_contract import parse_action
from h164.ood_gpu_infer import (
    SOURCE_ZIP_SHA,PARENT_LORA_SHA
)

PRED="h164-private-two-new-ood-predictions.json"
LOCK="H164_TRUE_NEW_APP_NATIVE_ONE_SHOT_CONSUMED.lock"
OUTPUT="H164_TRUE_NEW_APP_MODEL_NATIVE_REWARD.json"
SOURCE_SHA="3991d8b5b200ef9e80c5567534a6637df3df3319686ae9fe2c05b85cd7e75fd4"

def preflight(owner,model_root):
    owner=Path(owner)
    frozen_bytes=(owner/PRIVATE/"manifest.json").read_bytes()
    frozen=json.loads(frozen_bytes)
    qa=json.loads((owner/"private-h164/H164_OOD_64_PIXEL_QA.json").read_text("utf8"))
    if (digest(frozen_bytes)!=SOURCE_SHA or
        frozen.get("schema")!=SCHEMA or frozen.get("git_source_shas")!=PINS or
        qa.get("status")!="H164_OOD_64_NEW_APP_EXACT_BROWSER_PIXEL_REPLAY"):
        raise ValueError("H164 original new app source not independently frozen")
    expected=cases()
    records=frozen.get("private_cases")
    if not isinstance(records,list) or len(records)!=64:
        raise ValueError("H164 original Pell/Pikaday source count changed")
    for case,row in zip(expected,records):
        if any(row.get(k)!=v for k,v in case.items()):
            raise ValueError("H164 original OOD source scenario changed")
    prediction=json.loads((Path(model_root)/PRED).read_text("utf8"))
    if (prediction.get("status")!="H164_COMPLETED_128_REAL_GPU_OOD_CALLS"
        or prediction.get("schema")!=
        "hala.h164.two-new-original-app-free-t4-paired-eval.v1"
        or prediction.get("source_zip_sha256")!=SOURCE_ZIP_SHA
        or prediction.get("frozen_H161_adapter_sha256")!=PARENT_LORA_SHA
        or prediction.get("real_gpu_calls")!=128
        or prediction.get("optimizer_steps")!=0
        or prediction.get("paid_GPU") is not False
        or prediction.get("private_owner_gold_on_gpu") is not False):
        raise ValueError("H164 real 128 GPU OOD role/model provenance not qualified")
    new_sha=prediction.get("new_H164_adapter_sha256")
    if (not isinstance(new_sha,str) or len(new_sha)!=64
        or new_sha==PARENT_LORA_SHA):
        raise ValueError("H164 no new genuine updated model weights")
    rows=prediction.get("private_predictions")
    expected_pairs={(c["image_sha256"],role)
                    for c in records for role in ("before","after")}
    models={}
    if not isinstance(rows,list) or len(rows)!=128:
        raise ValueError("H164 exact 64×2 model OOD GPU calls missing")
    for row in rows:
        key=(row.get("image_sha"),row.get("role"))
        if (not isinstance(row,dict)
            or set(row)!={"role","image_sha","app","task",
                          "language","raw","parsed"}
            or key not in expected_pairs or key in models or
            row["parsed"]!=parse_action(row["raw"])):
            raise ValueError("H164 invalid/repeated model-native OOD assignment")
        models[key]=row
    if set(models)!=expected_pairs:
        raise ValueError("H164 OOD source-image × role paired GPU missing")
    for c in records:
        for role in ("before","after"):
            row=models[(c["image_sha256"],role)]
            if (row["app"]!=c["app"] or row["language"]!=c["language"]
                or row["task"]!=c["task"]):
                raise ValueError("H164 Pell/Pikaday model case mismatch")
    return records,models,new_sha

def native(browser,url,case,prediction):
    ctx=browser.new_context(viewport={"width":960,"height":720},
                            device_scale_factor=1,locale="en-US",
                            service_workers="block")
    try:
        page=ctx.new_page()
        if case["app"]=="pell":
            page.goto(url+"/demo.html",wait_until="domcontentloaded",timeout=15000)
            editor=page.locator(".pell-content")
            editor.wait_for(timeout=8000)
            message=f"Review OOD case {case['index']} now"
            editor.fill(message)
            editor.press("ControlOrMeta+A")
            expected_tag=TAGS[list(TAGS)[case["local_index"]%4]]
            earlier=editor.inner_html().lower()
        else:
            page.clock.install(time=datetime(2019,1,1,tzinfo=timezone.utc))
            page.route("**/*",lambda r:r.continue_()
                       if r.request.url.startswith(url) else r.abort())
            page.goto(url+"/index.html",wait_until="domcontentloaded",timeout=15000)
            page.wait_for_function(
                "() => typeof picker === 'object'",timeout=8000)
            field=page.locator("#datepicker")
            month=case["local_index"]//4
            day=8+(case["local_index"]%4)*3
            page.evaluate(
                "(date)=>picker.setDate(new Date(2019,date.month,date.day))",
                {"month":month,"day":day})
            earlier=field.input_value()
            field.click()
            target_day=day+1
        if digest(stable_png(page))!=case["image_sha256"]:
            raise ValueError("H164 frozen new app exact preaction PNG mismatch")
        proposed=prediction["parsed"]
        sent=proposed["valid"] and proposed["kind"]=="click"
        if sent:
            page.mouse.click(*proposed["points"][0])
            page.wait_for_timeout(80)
        if case["app"]=="pell":
            updated=editor.inner_html().lower()
            success=(updated!=earlier and message.lower() in updated and
                     (f"<{expected_tag}>" in updated or
                      (expected_tag=="strike" and "<s>" in updated)))
        else:
            success=(field.input_value()!=earlier and
                     page.evaluate("picker.getDate().getDate()")==target_day)
        return bool(sent and success),bool(sent)
    finally:ctx.close()

def run(owner,model_root):
    owner=Path(owner)
    cases,proposals,trained_sha=preflight(owner,model_root)
    originals=sources(owner)
    root=owner/"private-h164"
    consumed=root/LOCK
    result=root/OUTPUT
    if consumed.exists() or result.exists():
        raise FileExistsError("H164 truly new OOD real model browser gate one-shot consumed")
    with consumed.open("x",encoding="utf8") as f:
        f.write(json.dumps(
            {"status":"H164_TWO_NEW_ORIGINAL_APPS_NATIVE_ONE_SHOT_CONSUMED",
             "source_manifest_sha256":SOURCE_SHA,
             "trained_model_sha256":trained_sha,
             "h140_blind_calls":0},sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    trials=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            from contextlib import ExitStack
            with ExitStack() as stack:
                urls={app:stack.enter_context(serve(originals/app))
                      for app in ("pell","pikaday")}
                for case in cases:
                    for role in ("before","after"):
                        p=proposals[(case["image_sha256"],role)]
                        success,sent=native(browser,urls[case["app"]],case,p)
                        trials.append({
                            "image_sha":case["image_sha256"],
                            "role":role,"app":case["app"],"task":case["task"],
                            "language":case["language"],
                            "valid":p["parsed"]["valid"],
                            "click_sent":sent,
                            "real_native_original_app_success":success})
                    if len(trials)%16==0:
                        print("H164_REAL_TWO_NEW_APP_NATIVE_TRIALS",
                              len(trials),128,flush=True)
        finally:browser.close()
    by={(r["image_sha"],r["role"]):r for r in trials}
    summary=Counter();grouped=Counter()
    for c in cases:
        old=by[(c["image_sha256"],"before")]["real_native_original_app_success"]
        new=by[(c["image_sha256"],"after")]["real_native_original_app_success"]
        summary["before"]+=int(old);summary["after"]+=int(new)
        summary["wins"]+=int(new and not old)
        summary["regressions"]+=int(old and not new)
        grouped[c["app"]+"/before"]+=int(old)
        grouped[c["app"]+"/after"]+=int(new)
        grouped[c["language"]+"/before"]+=int(old)
        grouped[c["language"]+"/after"]+=int(new)
    dis=summary["wins"]+summary["regressions"]
    minimum=min(summary["wins"],summary["regressions"])
    p=min(1.,2*sum(math.comb(dis,k) for k in range(minimum+1))
          /(2**dis)) if dis else 1.
    receipt={
        "schema":"hala.h164.two-new-original-app-native-reward.v1",
        "status":"H164_COMPLETED_128_TRUE_NEW_APP_NATIVE_MODEL_EPISODES",
        "original_app_cases":64,"original_browser_trials":128,
        "baseline_success":summary["before"],
        "trained_success":summary["after"],
        "paired_wins":summary["wins"],
        "paired_regressions":summary["regressions"],
        "exact_paired_sign_p":p,"per_app_language":dict(grouped),
        "source_manifest_sha256":SOURCE_SHA,
        "trained_adapter_sha256":trained_sha,
        "owner_gold_sent_to_controller":False,
        "original_apps_not_in_train_or_dev":True,
        "one_shot_only":True,"h140_blind_calls":0,"paid_GPU":False,
        "private_rows":trials}
    result.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="private_rows"}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--model-root",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner,a.model_root),indent=2,sort_keys=True))

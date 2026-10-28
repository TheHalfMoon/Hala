"""H169 preregistered new same-family DEV: never train on these new scenes.

Original clean MIT upstream demos from H168, *new* 16..31 variants and
separate seed/layout/state; old H168 TRAIN captured indices 0..15.
The owner-only native success oracle stays off Kaggle and GitHub.
"""
import hashlib
import json
from pathlib import Path

from h168.source import verify_sources,historical_image_shas
from h168.capture import navigate
from h138.probe import serve
from h130.browser_env import _chromium_binary
from h159.stable_source_qa import stable_png

SCHEMA="hala.h169.original-app-new-bilingual-dev-32.v1"
COUNT=32

def sha(x):
    return hashlib.sha256(x).hexdigest()

def cases():
    rows=[]
    for app in ("choices","nouislider"):
        for n in range(16,32):
            rows.append({
                "index":169000+(0 if app=="choices" else 100)+n,
                "split":"DEV","app":app,"variant":n,
                "action_type":"type" if app=="choices" else "drag",
                "task":"add_new_tag" if app=="choices" else "drag_slider_handle",
                "language":"en" if n%2==0 else "ar"})
    assert len(rows)==COUNT and len({r["index"] for r in rows})==COUNT
    return rows

def admitted_history(owner,train_source):
    old=historical_image_shas(owner)
    train=json.loads((Path(train_source)/"manifest.json").read_text("utf8"))
    if train.get("schema")!="hala.h168.new-original-app-train-pilot.v1" or len(
            train.get("private_cases",[]))!=32:
        raise ValueError("H169 original prior TRAIN registration missing")
    old.update(x["image_sha256"] for x in train["private_cases"])
    return old

def run(owner,upstream,train_source,dest,mode):
    from playwright.sync_api import sync_playwright
    owner=Path(owner);upstream=Path(upstream);dest=Path(dest)
    proofs=verify_sources(upstream)
    previous=admitted_history(owner,train_source)
    plan=cases()
    if mode=="smoke":
        plan=[plan[0],plan[1],plan[16],plan[17]]
    if mode=="capture" and dest.exists():
        raise FileExistsError("H169 new original DEV can only be captured once")
    old=None
    if mode=="verify":
        old=json.loads((dest/"manifest.json").read_text("utf8"))
        if old.get("schema")!=SCHEMA or old.get("count")!=COUNT:
            raise ValueError("H169 frozen original DEV source changed")
    manifest={"schema":SCHEMA,"split":"DEV","count":len(plan),
              "reference_gold_uploaded":False,
              "previous_train_images_used":False,
              "new_variant_start":16,"new_variant_end":31,
              "upstreams":proofs,"private_cases":[]}
    references={"schema":"hala.h169.owner-only-native-dev-reference.v1",
                "owner_native_actions":[]}
    imgs={}
    with serve(upstream) as url:
        with sync_playwright() as playwright:
            opt={"headless":True}
            exe=_chromium_binary()
            if exe:opt["executable_path"]=exe
            browser=playwright.chromium.launch(**opt)
            try:
                for spec in plan:
                    picture,task,gold,succeeded=navigate(
                        browser,url,spec,execute=mode!="verify")
                    h=sha(picture)
                    if h in previous or h in imgs:
                        raise ValueError("H169 DEV old TRAIN/OOD or repeated screenshot hash")
                    row=dict(spec,image="images/"+str(spec["index"])+".png",
                             image_sha256=h,instruction=task,
                             native_reference_checked=(mode!="verify" and succeeded is True))
                    if mode=="verify":
                        expected=next((x for x in old["private_cases"]
                                       if x["index"]==spec["index"]),None)
                        if (not expected or expected["image_sha256"]!=h
                            or expected["instruction"]!=task
                            or (dest/row["image"]).read_bytes()!=picture):
                            raise ValueError("H169 DEV original Chromium replay mismatch")
                    else:
                        manifest["private_cases"].append(row)
                        references["owner_native_actions"].append(
                            dict(index=spec["index"],success=True,**gold))
                    imgs[spec["index"]]=picture
                    print("H169_ORIGINAL_DEV_"+mode.upper(),spec["app"],
                          spec["index"],h[:12],flush=True)
            finally:browser.close()
    if mode=="verify":
        if len(imgs)!=COUNT:
            raise ValueError("H169 full 32-case pixel replay incomplete")
        return {"status":"H169_ORIGINAL_NEW_DEV_PIXEL_EXACT_32_PASS",
                "cases":COUNT,"new_model_calls":0}
    if mode=="smoke":
        return {"status":"H169_ORIGINAL_NEW_DEV_SMOKE_PASS","cases":len(plan)}
    if len(imgs)!=COUNT or len(set(imgs))!=COUNT:
        raise ValueError("H169 only complete DEV accepted")
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"images").mkdir()
    for k,blob in imgs.items():
        (dest/"images"/(str(k)+".png")).write_bytes(blob)
    (dest/"manifest.json").write_text(json.dumps(
        manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    (dest/"owner-reference-private.json").write_text(json.dumps(
        references,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    return {"status":"H169_NEW_BILINGUAL_REAL_APP_DEV_32_CAPTURED",
            "cases":COUNT,"original_apps":2,"arabic":16,"english":16,
            "dev_only":True,"optimizer_steps":0,"owner_gold_uploaded":False,
            "prior_training_or_test_images_used":False}

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--owner",required=True)
    a.add_argument("--upstream",required=True)
    a.add_argument("--train-source",required=True)
    a.add_argument("--dest",required=True)
    a.add_argument("--mode",choices=("smoke","capture","verify"),required=True)
    x=a.parse_args()
    print(json.dumps(run(x.owner,x.upstream,x.train_source,x.dest,x.mode),
                     indent=2,sort_keys=True))

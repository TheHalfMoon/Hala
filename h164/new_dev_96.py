"""H164 independent 96 new DEV inputs from two genuine original GUI apps.

DEV owner-reference gold remains in an entirely separate local file; none of
its points or supervision fields are in the model input manifest or TRAIN.
"""
import argparse,hashlib,json
from pathlib import Path
from contextlib import ExitStack
from h130.browser_env import _chromium_binary
from h138.probe import serve
from h164.four_app_train_pilot import native_calculator,START
from h164.expand_288_train import (
    native_table,admitted_apps,prior,digest,SCHEMA as TRAIN_SCHEMA,PRIVATE as TRAIN_PRIVATE
)
from h164.protocol import validate_source_splits

SCHEMA="hala.h164.96-two-original-app-independent-dev.v1"
PRIVATE="private-h164/new-dev-96-v1"
PILOT=96
TASKS=("clear_selection","filter_name","sort_age")
CALC_TASKS=("press_plus","clear","calculate")

def cases():
    out=[]
    for app,ncases,tasks,base in (
        ("simple-table",48,TASKS,172001),
        ("calculator",48,CALC_TASKS,173001)
    ):
        for n in range(ncases):
            out.append({"split":"DEV","app":app,"task":tasks[n%3],
                        "language":"en" if n%2==0 else "ar",
                        "index":base+n,"local_index":PILOT+n})
    return out

def used(owner):
    owner=Path(owner)
    previous=prior(owner)
    train=owner/TRAIN_PRIVATE/"manifest.json"
    doc=json.loads(train.read_text("utf8"))
    proof=json.loads((owner/"private-h164/H164_EXPANDED_288_PIXEL_QA.json").read_text("utf8"))
    if (doc.get("schema")!=TRAIN_SCHEMA or len(doc["private_cases"])!=288 or
        proof.get("status")!="H164_288_NEW_TRAIN_EXACT_CHROMIUM_REPLAY"):
        raise ValueError("H164 DEV needs independently qualified 384 TRAIN scenes")
    previous.update(c["image_sha256"] for c in doc["private_cases"])
    return previous

def run(owner,mode):
    owner=Path(owner)
    roots=admitted_apps(owner)
    history=used(owner)
    destination=owner/PRIVATE
    if mode=="capture" and destination.exists():
        raise FileExistsError("H164 NEW DEV once-only source capture")
    doc=json.loads((destination/"manifest.json").read_text("utf8")) if mode=="verify" else None
    if doc and (doc.get("schema")!=SCHEMA or len(doc.get("private_cases",[]))!=96
                or doc.get("gold_model_inputs") is not False):
        raise ValueError("H164 independent DEV source manifest altered")
    sources=cases()
    if mode=="smoke":
        sources=[sources[i] for i in (0,1,2,48,49,50)]
    from playwright.sync_api import sync_playwright
    inputs=[];private_gold=[]
    with sync_playwright() as pw:
        options={"headless":True}
        exe=_chromium_binary()
        if exe:options["executable_path"]=exe
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            with ExitStack() as scope:
                urls={
                    app:scope.enter_context(serve(roots/app))
                    for app in ("simple-table","calculator")
                }
                if mode=="capture":
                    destination.mkdir(parents=True,exist_ok=False)
                    (destination/"images").mkdir()
                for i,case in enumerate(sources,1):
                    if case["app"]=="simple-table":
                        image,instruction,gold=native_table(
                            browser,urls["simple-table"]+"/examples/index.html",
                            case,execute=mode!="verify")
                    else:
                        tmp=dict(case,index=START+300+24+case["local_index"])
                        image,instruction,gold=native_calculator(
                            browser,urls["calculator"]+"/index.html",
                            tmp,execute=mode!="verify",rest_mouse=True)
                    sha=digest(image)
                    if sha in history:
                        raise ValueError("H164 DEV source screenshot overlaps TRAIN/consumed")
                    if mode=="capture":
                        if any(r["image_sha256"]==sha for r in inputs):
                            raise ValueError("H164 DEV duplicate screenshot")
                        fname=f'dev-{case["app"]}-{case["index"]}.png'
                        (destination/"images"/fname).write_bytes(image)
                        inputs.append({**case,"image":fname,
                                       "image_sha256":sha,
                                       "instruction":instruction,
                                       "action_type":gold["actions"][0]["kind"]})
                        private_gold.append({"image_sha256":sha,
                                             "native_reference_success":True,
                                             "owner_supervision":gold})
                    elif mode=="verify":
                        expected=doc["private_cases"][i-1]
                        if (any(expected.get(k)!=v for k,v in case.items()) or
                            expected["image_sha256"]!=sha or
                            (destination/"images"/expected["image"]).read_bytes()!=image):
                            raise ValueError("H164 independent DEV screenshot replay changed")
                    if i%12==0 or i==len(sources):
                        print("H164_DEV_"+mode.upper(),i,len(sources),case["app"],flush=True)
        finally:browser.close()
    if mode=="smoke":
        return {"status":"H164_SIX_NEW_DEV_ORIGINAL_NATIVE_SMOKE_PASS","model_calls":0}
    if mode=="verify":
        proof={"status":"H164_DEV_96_ORIGINAL_APP_EXACT_PIXEL_REPLAY",
               "source_cases":96,"reference_gold_gpu_inputs":False,
               "optimizer_steps":0,"model_calls":0,
               "h140_blind_calls":0,"paid_GPU":False}
        f=owner/"private-h164/H164_NEW_DEV_PIXEL_QA.json"
        if f.exists():raise FileExistsError("H164 DEV replay qualification already used")
        f.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n","utf8")
        return proof
    if len(inputs)!=96 or len({row["image_sha256"] for row in inputs})!=96:
        raise ValueError("H164 complete DEV screenshot split missing")
    manifest={"schema":SCHEMA,"dev_only":True,"gold_model_inputs":False,
              "cases":96,"train_data":False,
              "app_count":2,"optimizer_steps":0,"model_calls":0,
              "private_cases":inputs}
    src=(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    (destination/"manifest.json").write_bytes(src)
    (destination/"owner-gold-not-for-GPU.json").write_text(
        json.dumps({"source_manifest_sha256":digest(src),
                    "owner_reference_actions":private_gold},
                   ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    return {"status":"H164_96_NEW_INDEPENDENT_DEV_SCENES_CAPTURED",
            "manifest_sha256":digest(src),
            "owner_references_success":96,"DEV_gold_exported":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.mode),indent=2,sort_keys=True))

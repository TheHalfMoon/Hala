"""H162 one-shot real native original third-app exploratory transfer test."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import APPS,mouse_drag,serve
from h138.source import inspect_repo
from h159.stable_source_qa import stable_png
from h161.train_gpu import actions
from h162.gpu_infer import SOURCE_ZIP_SHA,BASE_SHA,TRAINED_SHA
from h162.original_third_app import (
    APP,PIN,SCHEMA,FIRST_ID,CASE_COUNT,state_for,sha,
)
from h162.pack_private_inference import SOURCE_SHA

PREDICTIONS="h162-private-third-app-predictions.json"
LOCK="H162_ORIGINAL_THIRD_APP_NEURAL_NATIVE_ONESHOT_CONSUMED.lock"
OUTPUT="H162_THIRD_APP_REAL_NEURAL_NATIVE_REWARD.json"


def preflight(owner,model_root,original):
    owner=Path(owner)
    frozen=json.loads((owner/"private-h162"/
                       "original-html5-board-source-once-v1"/
                       "manifest.json").read_text("utf8"))
    replay=json.loads((owner/"private-h162"/
                       "H162_SOURCE_PIXEL_QA.json").read_text("utf8"))
    if (frozen.get("schema")!=SCHEMA or frozen.get("upstream_sha")!=PIN or
        frozen.get("source_task_count")!=16 or
        replay.get("status")!="H162_16_EXACT_ORIGINAL_THIRD_APP_REPLAY_PASS"):
        raise ValueError("H162 owner third-app source independent SHA missing")
    source_path=owner/"private-h162"/"original-html5-board-source-once-v1"/"manifest.json"
    if sha(source_path.read_bytes())!=SOURCE_SHA:
        raise ValueError("H162 original app checksum mismatch")
    cases=frozen["private_cases"]
    if (len(cases)!=CASE_COUNT or
        len({x["image_sha256"] for x in cases})!=CASE_COUNT or
        [x["index"] for x in cases]!=list(range(FIRST_ID,FIRST_ID+16))):
        raise ValueError("H162 external source scenes no longer frozen")
    src=Path(original)
    if inspect_repo(src,PIN)!=PIN:
        raise ValueError("H162 third original app upstream SHA drift")
    predictions=json.loads((Path(model_root)/PREDICTIONS).read_text("utf8"))
    if (predictions.get("schema")!=
        "hala.h162.third-original-app-frozen-two-lora-gpu-inference.v1" or
        predictions.get("status")!=
        "H162_COMPLETED_32_REAL_FREE_T4_THIRD_APP_CALLS" or
        predictions.get("source_manifest_sha256")!=SOURCE_SHA or
        predictions.get("source_zip_sha256")!=SOURCE_ZIP_SHA or
        predictions.get("baseline_H157_lora_sha256")!=BASE_SHA or
        predictions.get("trained_H161_lora_sha256")!=TRAINED_SHA or
        predictions.get("gpu_inference_calls")!=32 or
        predictions.get("optimizer_steps")!=0 or
        predictions.get("h140_blind_calls")!=0 or
        predictions.get("paid_GPU") is not False):
        raise ValueError("H162 free GPU/adapter/case provenance not admissible")
    expected={(x["image_sha256"],role) for x in cases
              for role in ("before","after")}
    proposals={}
    records=predictions["private_predictions"]
    if not isinstance(records,list) or len(records)!=32:
        raise ValueError("H162 paired model predictions incomplete")
    for row in records:
        if (not isinstance(row,dict) or
            set(row)!={"role","image_sha","app","task","language","raw","parsed"} or
            (row["image_sha"],row["role"]) not in expected or
            (row["image_sha"],row["role"]) in proposals or
            row["parsed"]!=actions(row["raw"])):
            raise ValueError("H162 paired model action metadata does not match raw")
        proposals[(row["image_sha"],row["role"])]=row
    if set(proposals)!=expected:
        raise ValueError("H162 frozen image pair missing before real browser")
    for c in cases:
        for role in ("before","after"):
            row=proposals[(c["image_sha256"],role)]
            if (row["app"]!=c["app"] or row["task"]!=c["task"] or
                row["language"]!=c["language"]):
                raise ValueError("H162 paired model inference scenario misbound")
    return cases,proposals


def execute(browser,url,case,proposal):
    title,data=state_for(case)
    context=browser.new_context(viewport={"width":960,"height":720},
                                device_scale_factor=1,
                                service_workers="block",locale="en-US")
    context.add_init_script(
        "localStorage.setItem('kanban',"+
        json.dumps(json.dumps(data,ensure_ascii=False))+");")
    try:
        page=context.new_page()
        page.goto(url,wait_until="networkidle",timeout=20000)
        if title not in page.locator(".grid.grid-cols-4 > div").first.inner_text():
            raise ValueError("H162 original source task not displayed")
        if sha(stable_png(page))!=case["image_sha256"]:
            raise ValueError("H162 model trial screenshot identity changed")
        action=proposal["parsed"]
        sent=action["valid"] and action["kind"]=="drag"
        if sent:
            start,end=action["points"]
            mouse_drag(page,start,end)
        after=json.loads(page.evaluate("(key)=>localStorage.getItem(key)","kanban"))
        success=any(item.get("id")=="h162-"+str(case["index"])
                    for item in after[1]["cards"])
        return bool(sent and success),bool(sent)
    finally:
        context.close()


def run(owner,original,model_root):
    owner=Path(owner)
    cases,proposals=preflight(owner,model_root,original)
    root=owner/"private-h162"
    consumed=root/LOCK
    result=root/OUTPUT
    if consumed.exists() or result.exists():
        raise FileExistsError("H162 third external-app one-shot already consumed")
    with consumed.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "status":"H162_MODEL_EXTERNAL_BROWSER_ONESHOT_CONSUMED",
            "source_manifest_sha256":SOURCE_SHA,
            "before_model_sha":BASE_SHA,
            "after_model_sha":TRAINED_SHA,
            "h140_blind_calls":0
        },sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        chrome=_chromium_binary()
        if chrome:
            opts["executable_path"]=chrome
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(Path(original)/"dist") as url:
                for case in cases:
                    for role in ("before","after"):
                        proposal=proposals[(case["image_sha256"],role)]
                        won,sent=execute(browser,url,case,proposal)
                        rows.append({
                            "role":role,"image_sha":case["image_sha256"],
                            "language":case["language"],"valid":proposal["parsed"]["valid"],
                            "native_drag_sent":sent,
                            "real_original_app_task_success":won,
                        })
                    if len(rows)%8==0:
                        print("H162_REAL_EXTERNAL_HTML5_NATIVE_DRAG",
                              len(rows),32,flush=True)
        finally:
            browser.close()
    by={(r["image_sha"],r["role"]):r for r in rows}
    totals=Counter()
    lang=Counter()
    for c in cases:
        a=by[(c["image_sha256"],"after")]["real_original_app_task_success"]
        b=by[(c["image_sha256"],"before")]["real_original_app_task_success"]
        totals["before"]+=int(b)
        totals["after"]+=int(a)
        totals["new_wins"]+=int(a and not b)
        totals["regressions"]+=int(b and not a)
        lang[c["language"]+"/before"]+=int(b)
        lang[c["language"]+"/after"]+=int(a)
    discord=totals["new_wins"]+totals["regressions"]
    m=min(totals["new_wins"],totals["regressions"])
    p=min(1.,2*sum(math.comb(discord,k) for k in range(m+1))
          /(2**discord)) if discord else 1.
    report={
        "status":"H162_COMPLETED_32_REAL_ORIGINAL_THIRD_APP_MODEL_TASKS",
        "native_browser_trials":32,"original_app_cases":16,
        "before_actual_app_success":totals["before"],
        "after_actual_app_success":totals["after"],
        "paired_new_wins":totals["new_wins"],
        "paired_regressions":totals["regressions"],
        "exact_paired_sign_p":p,
        "per_language":dict(lang),
        "app_not_used_in_H161_training":True,
        "historically_seen_app_NOT_sealed_BLIND":True,
        "h140_blind_calls":0,"paid_GPU":False,
        "Hala35B_unchanged":True,"private_rows":rows,
    }
    result.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in report.items() if k!="private_rows"}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--owner",required=True)
    ap.add_argument("--original",required=True)
    ap.add_argument("--model-root",required=True)
    args=ap.parse_args()
    print(json.dumps(run(args.owner,args.original,args.model_root),
                     sort_keys=True,indent=2))

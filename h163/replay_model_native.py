"""H163 one-shot REAL native GridStack task success from frozen model clicks.

Never pass gold/DOM selectors to the model action controller. Owner-only original
app reference and DOM are used AFTER clicking for measuring actual task success.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import serve
from h159.stable_source_qa import stable_png
from h161.train_gpu import actions
from h163.source import admit, sha, SCHEMA
from h163.pack_private_inference import SOURCE_SHA,OLDER_MODEL_SHA,TRAINED_MODEL_SHA
from h163.gpu_infer import FROZEN_SOURCE_ZIP_SHA

LOCK="H163_BRAND_NEW_ORIGINAL_APP_MODEL_ONESHOT_CONSUMED.lock"
RESULT="H163_REAL_NEW_GRIDSTACK_BROWSER_MODEL_REWARD.json"
PRED="h163-new-original-app-predictions.json"


def preflight(owner,original,model_dir):
    owner=Path(owner)
    original=admit(original,owner)
    root=owner/"private-h163"
    frozen_raw=(root/"source-once-v1"/"manifest.json").read_bytes()
    source=json.loads(frozen_raw)
    proof=json.loads((root/"H163_SOURCE_PIXEL_QA.json").read_text("utf8"))
    if (sha(frozen_raw)!=SOURCE_SHA or source.get("schema")!=SCHEMA
        or proof.get("status")!="H163_16_OF_16_ORIGINAL_NEW_APP_EXACT_PIXEL_REPLAY_PASS"):
        raise ValueError("H163 new untouched original app source not qualified")
    cases=source.get("private_cases")
    if not isinstance(cases,list) or len(cases)!=16:
        raise ValueError("H163 original benchmark count changed")
    parent=Path(model_dir)
    predictions=json.loads((parent/PRED).read_text("utf8"))
    gpu_receipt=json.loads((owner/"private-h163"/"kaggle-output-v1"/
                            "h163-free-t4-original-app-receipt.json").read_text("utf8"))
    if (predictions.get("schema")!=
        "hala.h163.new-original-gridstack-real-paired-gpu.v1"
        or predictions.get("status")!=
        "H163_COMPLETE_32_REAL_FREE_GPU_INFERENCES"
        or predictions.get("original_app_source_manifest_sha256")!=SOURCE_SHA
        or predictions.get("original_app_source_zip_sha256")!=FROZEN_SOURCE_ZIP_SHA
        or predictions.get("baseline_adapter_sha256")!=OLDER_MODEL_SHA
        or predictions.get("trained_adapter_sha256")!=TRAINED_MODEL_SHA
        or predictions.get("gpu_calls")!=32
        or predictions.get("optimizer_steps")!=0
        or predictions.get("h140_blind_calls")!=0
        or predictions.get("paid_GPU") is not False
        or gpu_receipt.get("status")!=
        "H163_COMPLETED_32_REAL_FROZEN_GPU_CALLS"
        or gpu_receipt.get("gpu_calls")!=32
        or gpu_receipt.get("paid_GPU") is not False):
        raise ValueError("H163 owner original app frozen GPU provenance invalid")
    rows=predictions.get("private_predictions")
    expected={(c["image_sha256"],role) for c in cases
              for role in ("before","after")}
    proposals={}
    if not isinstance(rows,list) or len(rows)!=32 or len(expected)!=32:
        raise ValueError("H163 paired model role count incomplete")
    for row in rows:
        key=(row.get("image_sha"),row.get("role"))
        if (not isinstance(row,dict)
            or set(row)!={"role","image_sha","app","task","language","raw","parsed"}
            or key not in expected or key in proposals
            or actions(row["raw"])!=row["parsed"]):
            raise ValueError("H163 model proposal tampered/duplicated")
        proposals[key]=row
    if set(proposals)!=expected:
        raise ValueError("H163 missing frozen image/model paired inference")
    for case in cases:
        for role in ("before","after"):
            r=proposals[(case["image_sha256"],role)]
            if (r["app"]!=case["app"] or r["task"]!=case["task"]
                or r["language"]!=case["language"]):
                raise ValueError("H163 original source task/model assignment changed")
    return original,cases,proposals


def trial(browser,url,case,proposal):
    context=browser.new_context(
        viewport={"width":960,"height":720},device_scale_factor=1,
        locale="en-US",service_workers="block")
    try:
        page=context.new_page()
        page.route("https://**/*",lambda route:route.abort())
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(url,wait_until="load",timeout=20000)
        page.wait_for_function(
            "() => typeof GridStack==='function' && "
            "typeof grid!=='undefined' && grid.opts.column===12",timeout=10000)
        chosen=('\"grid.removeAll().load(list)\"'
                if case["layout"]=="list"
                else '\"grid.removeAll().load(test1)\"')
        page.locator("a[onclick="+chosen+"]").click()
        page.locator(f'a[onclick="column({case["pre_column"]})"]').click()
        page.wait_for_function("(n)=>grid.opts.column===n",
                               arg=case["pre_column"],timeout=5000)
        if errors:
            raise ValueError("H163 real original source app script errored")
        if sha(stable_png(page))!=case["image_sha256"]:
            raise ValueError("H163 before-action exact original PNG guard failed")
        act=proposal["parsed"]
        sent=act["valid"] and act["kind"]=="click"
        if case["task"]=="add_widget":
            earlier=int(page.evaluate("() => grid.save().length"))
        else:
            earlier=None
        if sent:
            page.mouse.click(*act["points"][0])
            page.wait_for_timeout(80)
        if case["task"]=="set_column":
            success=(page.evaluate("() => grid.opts.column")==
                     case["target_column"] and
                     page.locator("#column-text").inner_text().strip()==
                     str(case["target_column"]))
        else:
            success=(int(page.evaluate("() => grid.save().length"))==earlier+1)
        return bool(sent and success),bool(sent)
    finally:
        context.close()


def run(owner,original,model_dir):
    owner=Path(owner)
    original,cases,proposals=preflight(owner,original,model_dir)
    root=owner/"private-h163"
    used=root/LOCK
    result=root/RESULT
    if used.exists() or result.exists():
        raise FileExistsError("H163 new-app model-native one-shot already consumed")
    with used.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "status":"H163_NEW_REAL_APP_NEURAL_MODEL_EVAL_CONSUMED",
            "source_manifest_sha256":SOURCE_SHA,
            "frozen_before_sha":OLDER_MODEL_SHA,
            "frozen_after_sha":TRAINED_MODEL_SHA,
            "h140_blind_calls":0
        },sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    trials=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        executable=_chromium_binary()
        if executable:
            opts["executable_path"]=executable
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as base:
                url=base+"/demo/column.html"
                for case in cases:
                    for role in ("before","after"):
                        proposal=proposals[(case["image_sha256"],role)]
                        success,sent=trial(browser,url,case,proposal)
                        trials.append({
                            "image_sha":case["image_sha256"],
                            "role":role,"task":case["task"],
                            "language":case["language"],
                            "valid":proposal["parsed"]["valid"],
                            "click_sent":sent,"actual_original_app_success":success})
                    if len(trials)%8==0:
                        print("H163_REAL_NEW_APP_NATIVE_MODEL_CLICKS",
                              len(trials),32,flush=True)
        finally:
            browser.close()
    lookup={(r["image_sha"],r["role"]):r for r in trials}
    score=Counter()
    family=Counter()
    for c in cases:
        old=lookup[(c["image_sha256"],"before")]["actual_original_app_success"]
        new=lookup[(c["image_sha256"],"after")]["actual_original_app_success"]
        score["before"]+=int(old)
        score["after"]+=int(new)
        score["gains"]+=int(new and not old)
        score["regressions"]+=int(old and not new)
        family[c["task"]+"/before"]+=int(old)
        family[c["task"]+"/after"]+=int(new)
        family[c["language"]+"/before"]+=int(old)
        family[c["language"]+"/after"]+=int(new)
    discordant=score["gains"]+score["regressions"]
    minwins=min(score["gains"],score["regressions"])
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(minwins+1))
          /2**discordant) if discordant else 1.
    report={
        "status":"H163_NEW_ORIGINAL_GRIDSTACK_APP_REAL_NEURAL_REWARD_COMPLETE",
        "original_app_cases":16,"actual_browser_trials":32,
        "baseline_success":score["before"],
        "trained_success":score["after"],
        "paired_gains":score["gains"],
        "paired_regressions":score["regressions"],
        "exact_paired_sign_p":p,"by_task_and_language":dict(family),
        "new_upstream_original_app_not_in_Hala_training":True,
        "one_app_family_only":True,
        "h140_blind_calls":0,"paid_GPU":False,
        "flagship_Hala35B_unchanged":True,
        "private_rows":trials
    }
    result.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in report.items() if k!="private_rows"}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--original",required=True)
    p.add_argument("--model-root",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.original,args.model_root),
                     indent=2,sort_keys=True))

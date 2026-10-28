"""One-shot owner Windows native browser reward on frozen H172 new original DEV.

No reference point, locator or gold passed to frozen model. Score only real
original application state after model-generated screen-coordinate click.
"""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
from h130.browser_env import _chromium_binary
from h138.probe import serve
from h159.stable_source_qa import stable_png
from h164.native_action_contract import parse_action
from h171.source import verify_sources
from h171.image_integrity import same_exact_png_pixels
from h172.source_dev import SCHEMA,NODE_PREFIXES,cases
from h172.infer_gpu import DEV_ZIP_SHA,VISION_WEIGHTS_SHA

LOCK="H172_FRESH_NATIVE_DEV_ONE_SHOT_CONSUMED.lock"
OUTPUT_SCHEMA="hala.h172.new-november-node-dev-owner-native-reward.v1"
PRED_SCHEMA="hala.h172.frozen-paired-original-dev-predictions.v1"

def sha(blob):return hashlib.sha256(blob).hexdigest()

def admit_predictions(source,predictions):
    source=Path(source)
    doc=json.loads((source/"manifest.json").read_text("utf8"))
    rows=doc.get("private_cases")
    frozen=cases()
    if (doc.get("schema")!=SCHEMA or doc.get("split")!="DEV"
        or doc.get("count")!=32 or doc.get("owner_gold_uploaded") is not False
        or doc.get("previous_consumed_sources_excluded") is not True
        or not isinstance(rows,list) or len(rows)!=32):
        raise ValueError("H172 original new DEV source provenance invalid")
    for row,spec in zip(rows,frozen):
        if (any(row.get(k)!=v for k,v in spec.items())
            or row.get("owner_native_reference_passed") is not True
            or row.get("image")!="images/"+str(row["index"])+".png"
            or sha((source/row["image"]).read_bytes())!=row["image_sha256"]):
            raise ValueError("H172 original owner snapshot/scene integrity invalid")
    report=json.loads(Path(predictions).read_text("utf8"))
    if (report.get("schema")!=PRED_SCHEMA
        or report.get("status")!="H172_PRIVATE_64_REAL_GPU_DEV_PREDICTIONS_COMPLETE"
        or report.get("source_dev_zip_sha256")!=DEV_ZIP_SHA
        or report.get("trained_vision_sha256")!=VISION_WEIGHTS_SHA
        or report.get("real_gpu_calls")!=64
        or report.get("paired_dev_cases")!=32
        or report.get("optimizer_steps")!=0
        or report.get("owner_gold_on_gpu") is not False
        or report.get("previous_consumed_ood_used") is not False
        or report.get("paid_gpu") is not False):
        raise ValueError("H172 frozen paired private GPU proposals unqualified")
    proposals=report.get("predictions")
    if not isinstance(proposals,list) or len(proposals)!=64:
        raise ValueError("H172 all paired GPU outputs must exist before native replay")
    result={}
    expected={(x["index"],role) for x in frozen for role in ("before","after")}
    for proposal in proposals:
        key=(proposal.get("case_index"),proposal.get("role"))
        scene=next((x for x in rows if x["index"]==proposal.get("case_index")),None)
        if (key not in expected or key in result or scene is None
            or proposal.get("image_sha256")!=scene["image_sha256"]
            or proposal.get("app")!=scene["app"]
            or proposal.get("language")!=scene["language"]
            or proposal.get("action_type")!="click"
            or proposal.get("parsed")!=parse_action(proposal.get("raw"))):
            raise ValueError("H172 repeated, tampered or misassigned frozen action")
        result[key]=proposal
    if set(result)!=expected:
        raise ValueError("H172 frozen before/after 32 original DEV cases incomplete")
    return rows,result

def replay_original(browser,url,row,proposal,source):
    context=browser.new_context(viewport={"width":960,"height":720},
        device_scale_factor=1,locale="en-US",service_workers="block")
    try:
        page=context.new_page()
        page.route("**/*",lambda route:route.continue_()
            if route.request.url.startswith(url) else route.abort())
        n=row["variant"]
        if row["app"]=="flatpickr":
            target="flatpickr-stage/package/index.template.html"
        else:
            target="awesomplete-upstream/index.html"
        response=page.goto(url+target,wait_until="load",timeout=25000)
        if not response or response.status!=200:
            raise ValueError("H172 original source app HTTP failed")
        if row["app"]=="flatpickr":
            page.wait_for_function("()=>typeof fp!=='undefined' && !!fp.setDate",
                                    timeout=8000)
            seed=5+n;desired=6+n
            page.evaluate("(d)=>fp.setDate(new Date(2026,10,d),true)",seed)
            field=page.locator("input.date")
            field.click()
            target_label=f"2026-11-{desired:02}"
            target=page.locator(
                f'.flatpickr-day[aria-label="November {desired}, 2026"]')
            target.wait_for(timeout=8000)
            target.scroll_into_view_if_needed()
        else:
            field=page.locator("input.awesomplete").first
            field.wait_for(timeout=9000)
            field.scroll_into_view_if_needed()
            field.fill(NODE_PREFIXES[n])
            target=page.locator("div.awesomplete li").filter(
                has_text="Node.js").first
            target.wait_for(timeout=9000)
            target.scroll_into_view_if_needed()
            target_label="Node.js"
        original=(Path(source)/row["image"]).read_bytes()
        if not same_exact_png_pixels(original,stable_png(page)):
            raise ValueError("H172 owner new DEV preregistered original pixel replay drift")
        action=proposal["parsed"]
        valid=action["valid"] is True and action["kind"]=="click"
        if valid:
            point=action["points"][0]
            if (type(point)!=list or len(point)!=2 or
                not 0<=point[0]<960 or not 0<=point[1]<720):
                valid=False
            else:
                page.mouse.click(*point)
                page.wait_for_timeout(80)
        success=valid and field.input_value()==target_label
        return {"success":bool(success),"valid_click":bool(valid),
                "parsed_kind":action.get("kind")}
    finally:
        context.close()

def run(owner,stage,flat,awesome,tarball,source,predictions,output):
    from playwright.sync_api import sync_playwright
    owner=Path(owner);output=Path(output)
    lock=owner/"private-h172"/LOCK
    if output.exists() or lock.exists():
        raise ValueError("H172 one-shot original new DEV already evaluated/consumed")
    verify_sources(flat,awesome,stage,tarball)
    scenes,proposals=admit_predictions(source,predictions)
    details=[];counts=Counter()
    with serve(Path(stage)) as url:
        with sync_playwright() as pw:
            opts={"headless":True}
            exe=_chromium_binary()
            if exe:opts["executable_path"]=exe
            browser=pw.chromium.launch(**opts)
            try:
                for row in scenes:
                    for role in ("before","after"):
                        observed=replay_original(
                            browser,url,row,proposals[(row["index"],role)],source)
                        counts[role+"/calls"]+=1
                        counts[role+"/success"]+=int(observed["success"])
                        counts[role+"/"+row["app"]+"/success"]+=int(observed["success"])
                        counts[role+"/"+row["language"]+"/success"]+=int(observed["success"])
                        details.append(dict(index=row["index"],role=role,
                            app=row["app"],language=row["language"],
                            image_sha256=row["image_sha256"],**observed))
                        print("H172_NATIVE_NEW_DEV",len(details),64,role,
                              observed["success"],flush=True)
            finally:browser.close()
    pair={(r["index"],r["role"]):r["success"] for r in details}
    wins=sum(pair[(x["index"],"after")] and not pair[(x["index"],"before")]
             for x in scenes)
    losses=sum(pair[(x["index"],"before")] and not pair[(x["index"],"after")]
               for x in scenes)
    if len(details)!=64 or counts["before/calls"]!=32 or counts["after/calls"]!=32:
        raise ValueError("H172 frozen new DEV original native actions incomplete")
    receipt=dict(schema=OUTPUT_SCHEMA,
        status="H172_FRESH_ORIGINAL_DEV_NATIVE_ONE_SHOT_CONSUMED",
        cases=32,real_gpu_proposals=64,
        before_success=counts["before/success"],
        after_success=counts["after/success"],
        paired_wins=wins,paired_regressions=losses,
        counts=dict(counts),private_outcomes=details,
        source_dev_zip_sha256=DEV_ZIP_SHA,
        trained_vision_sha256=VISION_WEIGHTS_SHA,
        real_original_browser_native_reward=True,
        owner_gold_on_gpu=False,previous_consumed_ood_used=False,
        new_dev_evaluated_once=True,paid_gpu=False,
        optimizer_steps=0,flagship_35b_unchanged=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps({
        "status":"H172_FRESH_DEV_PERMANENTLY_CONSUMED",
        "source_zip_sha256":DEV_ZIP_SHA},sort_keys=True)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="private_outcomes"}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("owner","stage","flat","awesome","tarball",
                "source","predictions","output"):
        parser.add_argument("--"+key,required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.owner,a.stage,a.flat,a.awesome,a.tarball,
        a.source,a.predictions,a.output),indent=2,sort_keys=True))

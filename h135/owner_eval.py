"""Owner-only after-inference real Chromium evaluation for H135 pilot.

No training and no injected correct pixels. Frozen native model outputs drive
the real mouse. Actual DOM bounds used for scoring only after prediction.
"""
import json
import math
import re
import statistics
from pathlib import Path

from h135.source import check,sha
from h135.ablate_gpu import SCHEMA_OUT,MODES,MODELS
from h132.grounder_contract import MODEL_ID,MODEL_SHA,native_point
from h134.sealed_gpu import TRAINED_ADAPTER_SHA
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h131.owner_diagnostics import distance_to_box
from h133.paired_stats import exact_discordant_p

def validate_raw(source_dir,predictions_dir):
    manifest=check(source_dir,"pilot")
    data=json.loads((Path(predictions_dir)/"private-pilot-score.json").read_text("utf8"))
    raw=json.loads((Path(predictions_dir)/"private-raw-responses.json").read_text("utf8"))
    if (data.get("schema")!=SCHEMA_OUT or data.get("status")!="H135_REAL_PRIVATE_PILOT12_RESOLUTION_PAIRED" or
        data.get("pilot_manifest_sha256")!=sha((Path(source_dir)/"manifest.json").read_bytes()) or
        data.get("base_id")!=MODEL_ID or data.get("base_revision")!=MODEL_SHA or
        data.get("adapter_sha256")!=TRAINED_ADAPTER_SHA or
        data.get("actual_neural_calls")!=96 or
        data.get("actual_optimizer_steps")!=0 or
        data.get("no_sealed_holdout_attached") is not True):
        raise ValueError("H135 frozen source/real neural inference provenance invalid")
    if not isinstance(raw,list) or len(raw)!=96:
        raise ValueError("H135 strict two models x two modes x12 x2 predictions")
    indexed={}
    k=0
    for arm in MODELS:
        for mode in MODES:
            for case in manifest["cases"]:
                for role in ("start","end"):
                    r=raw[k];k+=1
                    if (r.get("arm")!=arm or r.get("resolution")!=mode or
                        r.get("id")!=case["id"] or
                        r.get("image_sha256")!=case["image_sha256"] or
                        r.get("role")!=role):
                        raise ValueError("H135 pilot model arm/scene/role mismatch")
                    point=native_point(r.get("raw"),WIDTH,HEIGHT)
                    if r.get("point")!=point:
                        raise ValueError("H135 model native raw response does not match point")
                    b=case[role+"_bbox"]
                    hit=(point is not None and
                         b["x"]<=point[0]<=b["x"]+b["width"] and
                         b["y"]<=point[1]<=b["y"]+b["height"])
                    if r.get("hit") is not bool(hit):
                        raise ValueError("H135 claimed point hit not supported by true browser bounding box")
                    indexed[(arm,mode,case["id"],role)]=point
    return manifest,indexed

def evaluate(source_dir,predictions_dir,output_file):
    manifest,pts=validate_raw(source_dir,predictions_dir)
    source=Path(__file__).with_name("fixture.html").resolve()
    from playwright.sync_api import sync_playwright
    results={}
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts)
        try:
            context=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                        device_scale_factor=1,service_workers="block")
            context.route(re.compile(r"^https?://"),lambda r:r.abort())
            page=context.new_page()
            for arm in MODELS:
                for mode in MODES:
                    rows=[]
                    for case in manifest["cases"]:
                        seed=case["id"].rsplit("-",1)[1]
                        page.goto(source.as_uri()+"?id="+seed,wait_until="load")
                        page.locator('[data-role="handle"]').wait_for()
                        before=page.screenshot(type="png")
                        if sha(before)!=case["image_sha256"]:
                            raise ValueError("H135 frozen actual screenshot/owner Chrome changed")
                        pstart=pts[(arm,mode,case["id"],"start")]
                        pend=pts[(arm,mode,case["id"],"end")]
                        err_start=(distance_to_box(pstart,case["start_bbox"]) if pstart else None)
                        err_end=(distance_to_box(pend,case["end_bbox"]) if pend else None)
                        started=err_start==0 if err_start is not None else False
                        ended=err_end==0 if err_end is not None else False
                        if pstart is None or pend is None:
                            success=changed=False
                        else:
                            page.mouse.move(*pstart);page.mouse.down()
                            page.mouse.move(*pend,steps=12);page.mouse.up()
                            success=page.locator("#canvas").get_attribute("data-success")=="true"
                            changed=sha(page.screenshot(type="png"))!=sha(before)
                        rows.append({
                            "case_id":case["id"],"start_hit":started,
                            "end_hit":ended,"start_error_px":round(err_start,2) if err_start is not None else None,
                            "end_error_px":round(err_end,2) if err_end is not None else None,
                            "task_completed":success,"scene_changed":changed})
                    start_err=[x["start_error_px"] for x in rows if x["start_error_px"] is not None]
                    results[arm+"__"+mode]={
                        "n":len(rows),"start_hits":sum(x["start_hit"] for x in rows),
                        "end_hits":sum(x["end_hit"] for x in rows),
                        "real_gui_successes":sum(x["task_completed"] for x in rows),
                        "visible_changes":sum(x["scene_changed"] for x in rows),
                        "median_start_error_px":statistics.median(start_err) if start_err else None,
                        "rows":rows}
            context.close()
        finally:browser.close()
    paired={}
    for arm in MODELS:
        low=results[arm+"__downsample_640"]["rows"]
        high=results[arm+"__native_960"]["rows"]
        a={x["case_id"]:x for x in low};b={x["case_id"]:x for x in high}
        if len(a)!=12 or len(b)!=12 or set(a)!=set(b):
            raise ValueError("H135 mandatory matched pair scenes absent")
        new_only=sum(b[k]["task_completed"] and not a[k]["task_completed"] for k in a)
        old_only=sum(not b[k]["task_completed"] and a[k]["task_completed"] for k in a)
        paired[arm]={
            "native_only_GUI":new_only,"small_only_GUI":old_only,
            "exact_two_sided_p":exact_discordant_p(new_only,old_only),
            "n":12,"generalized_improvement_proven":False}
    output={
        "schema":"hala.h135.owner-private-real-browser-resolution-ablation.v1",
        "status":"H135_PILOT_PRIVATE_96_NEURAL_AND_REAL_BROWSER_SCORED",
        "frozen_pilot_manifest_sha256":sha((Path(source_dir)/"manifest.json").read_bytes()),
        "real_neural_calls":96,"real_browser_attempts":48,
        "models":results,"paired":paired,
        "untouched_H135_sealed_holdout":True,
        "trained_weights_modified":False,"release_qualified":False}
    output_file=Path(output_file)
    if output_file.exists():raise ValueError("H135 owner pilot score create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n","utf8")
    return output

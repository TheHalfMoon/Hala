"""H135 owner post-inference ONE-TIME sealed18 real browser replay.

A frozen pilot-selected visual resolution and adapter are read from the
private Kaggle sealed receipt. No gold/DOM used in action generation.
"""
import json
import re
from pathlib import Path
from h135.source import check,sha
from h135.sealed_gpu import (FORMAT,SEALED_MANIFEST_SHA,TRAINED_ADAPTER_SHA,
                             SELECTED_MODE,SELECTED_DECISION_SHA)
from h132.grounder_contract import MODEL_ID,MODEL_SHA,native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h133.paired_stats import exact_discordant_p

def verify(source,result,points):
    if (result.get("schema")!=FORMAT or
        result.get("status")!="H135_PRIVATE_ONCE_SEALED18_72_NEURAL_SCORED" or
        result.get("selected_resolution_mode")!=SELECTED_MODE or
        result.get("frozen_pilot_decision_sha256")!=SELECTED_DECISION_SHA or
        result.get("sealed_manifest_sha256")!=SEALED_MANIFEST_SHA or
        result.get("immutable_adapter_sha256")!=TRAINED_ADAPTER_SHA or
        result.get("frozen_base_id")!=MODEL_ID or
        result.get("frozen_base_revision")!=MODEL_SHA or
        result.get("actual_neural_calls")!=72 or
        result.get("optimizer_steps")!=0 or
        result.get("no_pilot_data_attached") is not True):
        raise ValueError("H135 sealed independent pretrained model evidence drift")
    if type(points) is not list or len(points)!=72:
        raise ValueError("H135 sealed exact 2 x18 x2 real neural calls required")
    lookup={}
    i=0
    for arm in ("pretrained","h134_adapter"):
        for case in source["cases"]:
            for role in ("start","end"):
                row=points[i];i+=1
                if (row.get("arm")!=arm or row.get("case_id")!=case["id"] or
                    row.get("image_sha256")!=case["image_sha256"] or
                    row.get("role")!=role):
                    raise ValueError("H135 sealed model raw scene identity mismatch")
                native=native_point(row.get("raw"),WIDTH,HEIGHT)
                if row.get("point")!=native:
                    raise ValueError("H135 sealed model native raw point edited")
                bbox=case[role+"_bbox"]
                hit=(native is not None and
                     bbox["x"]<=native[0]<=bbox["x"]+bbox["width"] and
                     bbox["y"]<=native[1]<=bbox["y"]+bbox["height"])
                if row.get("hit") is not bool(hit):
                    raise ValueError("H135 sealed raw bbox score not owner replicable")
                lookup[(arm,case["id"],role)]=native
    return lookup

def replay(source_dir,run_dir,output_path):
    source_dir,run_dir,output_path=map(Path,(source_dir,run_dir,output_path))
    source=check(source_dir,"sealed")
    if sha((source_dir/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H135 sealed holdout SHA changed post pilot")
    report=json.loads((run_dir/"private-sealed-metrics.json").read_text("utf8"))
    raw=json.loads((run_dir/"private-sealed-model-points.json").read_text("utf8"))
    predicted=verify(source,report,raw)
    outcomes={}
    from playwright.sync_api import sync_playwright
    fixture=Path(__file__).with_name("fixture.html").resolve()
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route(re.compile(r"^https?://"),lambda route:route.abort())
            page=ctx.new_page()
            for arm in ("pretrained","h134_adapter"):
                rows=[]
                for case in source["cases"]:
                    page.goto(fixture.as_uri()+"?id="+case["id"].split("-")[-1],wait_until="load")
                    page.locator('[data-role="handle"]').wait_for()
                    original=page.screenshot(type="png")
                    if sha(original)!=case["image_sha256"]:
                        raise ValueError("H135 owner sealed Chromium scene changed")
                    start=predicted[(arm,case["id"],"start")]
                    end=predicted[(arm,case["id"],"end")]
                    if start is None or end is None:
                        completed=False
                        changed=False
                    else:
                        page.mouse.move(*start);page.mouse.down()
                        page.mouse.move(*end,steps=12);page.mouse.up()
                        completed=page.locator("#canvas").get_attribute("data-success")=="true"
                        changed=sha(page.screenshot(type="png"))!=sha(original)
                    rows.append({"id":case["id"],"full_gui_success":completed,
                                 "state_change":changed})
                outcomes[arm]=rows
            ctx.close()
        finally:browser.close()
    base={r["id"]:r["full_gui_success"] for r in outcomes["pretrained"]}
    adapted={r["id"]:r["full_gui_success"] for r in outcomes["h134_adapter"]}
    only_adapter=sum(not base[k] and adapted[k] for k in base)
    only_base=sum(base[k] and not adapted[k] for k in base)
    result={
      "status":"H135_OWNER_ONCE_SEALED18_REAL_GUI_REPLAY_PASS",
      "frozen_selected_mode":SELECTED_MODE,"frozen_pilot_decision_sha256":SELECTED_DECISION_SHA,
      "owner_chromium_real_browser_actions":36,
      "sealed_original_scenes":18,"actual_neural_calls":72,
      "pretrained_full_gui_successes":sum(base.values()),
      "trained_h134_adapter_full_gui_successes":sum(adapted.values()),
      "paired_adapter_only_success":only_adapter,"paired_base_only_success":only_base,
      "exact_two_sided_p":exact_discordant_p(only_adapter,only_base),
      "source_and_raw_inference_verified":True,
      "new_optimizer_steps":0,"flagship_Hala_35B_improved_proven":False,
      "generalized_reliability_proven":False,
    }
    if output_path.exists():raise ValueError("H135 sealed owner result must be create-only")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
    return result

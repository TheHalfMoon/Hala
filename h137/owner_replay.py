"""H137 owner independent blind evaluation of real MIT-library mouse behaviour.

Replay unaltered neural START/END predictions on genuine locally vendored
noUiSlider, Split.js and SortableJS. The 36 owner gold gestures are solely
admission evidence and must NEVER be substituted for the model's actions.
"""
import json
import re
from pathlib import Path

from h137.source import SCHEMA,digest,check,IDS
from h137.gpu_eval import (
    OUTPUT_SCHEMA,FROZEN_SOURCE_MANIFEST_SHA,FROZEN_ADAPTER_SHA)
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h133.paired_stats import exact_discordant_p

ARMS=("pretrained","h136_64step_adapter")

def verify(source,report,raw):
    if (report.get("schema")!=OUTPUT_SCHEMA or
        report.get("status")!="H137_PRIVATE_THREE_REAL_MIT_INTERACTION_LIBRARIES_144_INFERENCE" or
        report.get("actual_neural_calls")!=144 or
        report.get("actual_optimizer_steps")!=0 or
        report.get("source_sha256")!=FROZEN_SOURCE_MANIFEST_SHA or
        report.get("frozen_adapter_sha256")!=FROZEN_ADAPTER_SHA or
        report.get("training_source_attached") is not False or
        report.get("paid_compute_used") is not False or
        report.get("benchmark_is_independent_production_apps") is not False):
        raise ValueError("H137 owner raw frozen neural/provenance receipt mismatch")
    if not isinstance(raw,list) or len(raw)!=144:raise ValueError("H137 144 raw predictions needed")
    indexed={}
    k=0
    for arm in ARMS:
        for case in source["cases"]:
            for role in ("start","end"):
                v=raw[k];k+=1
                if (v.get("arm")!=arm or v.get("case_id")!=case["case_id"] or
                    v.get("lang")!=case["lang"] or v.get("family")!=case["family"] or
                    v.get("image_sha256")!=case["image_sha256"] or
                    v.get("role")!=role):
                    raise ValueError("H137 real library scene/role/arm binding inconsistent")
                p=native_point(v.get("raw"),WIDTH,HEIGHT)
                if v.get("point")!=p:raise ValueError("H137 neural text native pixel point tampering")
                box=case[role+"_bbox"]
                hit=(p is not None and box["x"]<=p[0]<=box["x"]+box["width"]
                     and box["y"]<=p[1]<=box["y"]+box["height"])
                if v.get("hit") is not bool(hit):
                    raise ValueError("H137 claimed start/end box hit unverified")
                indexed[(arm,case["case_id"],role)]=p
    return indexed

def replay(source_root,result_root,output_file):
    from playwright.sync_api import sync_playwright
    source_root,result_root,output_file=map(Path,(source_root,result_root,output_file))
    manifest=check(source_root)
    if digest((source_root/"manifest.json").read_bytes())!=FROZEN_SOURCE_MANIFEST_SHA:
        raise ValueError("H137 owner-private original library screenshot manifest drift")
    result=json.loads((result_root/"private-h137-neural-results.json").read_text("utf8"))
    raw=json.loads((result_root/"private-h137-raw-neural.json").read_text("utf8"))
    prediction=verify(manifest,result,raw)
    fixture=source_root/"_runtime"/"fixture.html"
    if not fixture.exists():raise ValueError("H137 genuine private vendored UI runtime missing")
    original_assets=manifest["external_runtime_assets"]
    for asset,info in original_assets.items():
        if digest((fixture.parent/"vendor"/asset).read_bytes())!=info["sha256"]:
            raise ValueError("H137 genuine upstream JS runtime changed after source freeze")
    rows=[]
    actual_drags=0
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route(re.compile(r"^https?://"),lambda route:route.abort())
            page=ctx.new_page()
            for arm in ARMS:
                for case in manifest["cases"]:
                    seed=case["case_id"].split("-")[-1]
                    page.goto(fixture.as_uri()+"?id="+seed,wait_until="load")
                    page.locator("#start-target").wait_for()
                    if page.locator("#workspace").get_attribute("data-vendor")!=case["family"]:
                        raise ValueError("H137 vendor-provided interaction runtime absent")
                    if digest(page.screenshot(type="png"))!=case["image_sha256"]:
                        raise ValueError("H137 real independent library screenshot byte SHA drift")
                    start=prediction[(arm,case["case_id"],"start")]
                    end=prediction[(arm,case["case_id"],"end")]
                    if start is None or end is None:
                        worked=False
                    else:
                        actual_drags+=1
                        page.mouse.move(*start)
                        page.mouse.down()
                        page.mouse.move(*end,steps=24)
                        page.mouse.up()
                        page.wait_for_timeout(75)
                        worked=page.locator("#workspace").get_attribute("data-success")=="true"
                    rows.append({"arm":arm,"case_id":case["case_id"],
                                 "lang":case["lang"],"family":case["family"],
                                 "task_success":worked})
            ctx.close()
        finally:browser.close()
    before={r["case_id"]:r["task_success"] for r in rows if r["arm"]==ARMS[0]}
    after={r["case_id"]:r["task_success"] for r in rows if r["arm"]==ARMS[1]}
    trained_only=sum(after[k] and not before[k] for k in before)
    base_only=sum(before[k] and not after[k] for k in before)
    segments={}
    for case in manifest["cases"]:
        segment=case["family"]+"__"+case["lang"]
        sub=segments.setdefault(segment,{"cases":0,"pretrained_successes":0,"adapter_successes":0})
        sub["cases"]+=1
        sub["pretrained_successes"]+=int(before[case["case_id"]])
        sub["adapter_successes"]+=int(after[case["case_id"]])
    evidence={
        "schema":"hala.h137.owner-genuine-open-source-library-browser-replay.v1",
        "status":"H137_OWNER_THREE_REAL_MIT_LIBRARY_RUNTIME_REPLAY_PASS",
        "frozen_source_manifest_sha256":FROZEN_SOURCE_MANIFEST_SHA,
        "frozen_h136_adapter_sha256":FROZEN_ADAPTER_SHA,
        "genuine_vendor_engine_scenes":36,
        "english_scenes":18,"arabic_scenes":18,
        "genuine_vendor_libraries":3,
        "neural_calls":144,"evaluated_browser_scene_pairs":72,
        "actual_model_proposed_drag_attempts":actual_drags,
        "real_task_success_pretrained":sum(before.values()),
        "real_task_success_h136_adapter":sum(after.values()),
        "paired_adapter_only":trained_only,"paired_pretrained_only":base_only,
        "exact_two_sided_p":exact_discordant_p(trained_only,base_only),
        "per_library_per_language":segments,
        "genuinely_independent_production_applications_benchmarked":False,
        "Hala_flagship_35B_changed":False,
        "optimizer_steps":0,
        "generalized_real_application_ability_proven":False,
    }
    if output_file.exists():raise ValueError("H137 owner replay report create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n","utf8")
    return evidence

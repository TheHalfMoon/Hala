"""Owner Windows actual Playwright DEV24 full-drag verification.

DEV is development-only; never substitute this for SEALED24 generalization.
Generated model raw points are executed as-is with true mouse gestures.
"""
import json
import re
from pathlib import Path
from h136.source import check,sha
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

def evaluate(dev_dir,trained_dir,report_path):
    dev_dir,trained_dir,report_path=map(Path,(dev_dir,trained_dir,report_path))
    source=check(dev_dir,"dev")
    phases={
       "pretrained":json.loads((trained_dir/"private-dev-before.json").read_text("utf8")),
       "trained_adapter":json.loads((trained_dir/"private-dev-after.json").read_text("utf8"))}
    if any(len(raw)!=48 for raw in phases.values()):
        raise ValueError("H136 DEV24 real neural responses incomplete")
    predictions={}
    for phase,rows in phases.items():
        by_id={}
        for pair_idx,scene in enumerate(source["cases"]):
            pair=rows[pair_idx*2:pair_idx*2+2]
            if (pair[0].get("case_id")!=scene["id"] or
                pair[1].get("case_id")!=scene["id"] or
                pair[0].get("role")!="start" or pair[1].get("role")!="end"):
                raise ValueError("H136 DEV24 neural role/ID mismatch")
            points={}
            for r in pair:
                point=native_point(r.get("raw"),WIDTH,HEIGHT)
                if r.get("prediction")!=point:
                    raise ValueError("H136 native DEV point was modified")
                points[r["role"]]=point
            by_id[scene["id"]]=points
        predictions[phase]=by_id
    from playwright.sync_api import sync_playwright
    fixture=Path(__file__).with_name("fixture.html").resolve()
    outcomes={}
    attempts=0
    with sync_playwright() as p:
        opts={"headless":True}
        chromium=_chromium_binary()
        if chromium:opts["executable_path"]=chromium
        browser=p.chromium.launch(**opts)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route(re.compile(r"^https?://"),lambda req:req.abort())
            page=ctx.new_page()
            for phase in phases:
                result=[]
                for scene in source["cases"]:
                    page.goto(fixture.as_uri()+"?id="+scene["id"].rsplit("-",1)[1])
                    page.locator('[data-role="handle"]').wait_for()
                    if sha(page.screenshot(type="png"))!=scene["image_sha256"]:
                        raise ValueError("H136 real DEV browser screenshot SHA drift")
                    points=predictions[phase][scene["id"]]
                    if points["start"] is None or points["end"] is None:
                        success=False
                    else:
                        attempts+=1
                        page.mouse.move(*points["start"])
                        page.mouse.down()
                        page.mouse.move(*points["end"],steps=12)
                        page.mouse.up()
                        success=page.locator("#canvas").get_attribute("data-success")=="true"
                    result.append({"case":scene["id"],"success":success})
                outcomes[phase]=result
            ctx.close()
        finally:browser.close()
    res={
       "status":"H136_OWNER_ACTUAL_DEV24_MODEL_MOUSE_REPLAY",
       "dev_original_scenes":24,"evaluated_model_scene_pairs":48,
       "actual_drag_attempts":attempts,
       "pretrained_task_successes":sum(x["success"] for x in outcomes["pretrained"]),
       "trained_adapter_task_successes":sum(x["success"] for x in outcomes["trained_adapter"]),
       "sealed24_seen":False,
       "generalization_proven":False}
    if report_path.exists():raise ValueError("H136 DEV browser evidence create-only")
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(res,indent=2,sort_keys=True)+"\n","utf8")
    return res

"""H136 owner Windows replay of frozen sealed24 neural outputs on actual Chromium.

All neural responses are recorded before DOM/ground-truth is inspected.
Never modify model-generated start/end coordinates to reach a goal.
"""
import json
import re
from pathlib import Path
from h136.source import check,sha
from h136.sealed_gpu import FORMAT,SEALED_MANIFEST_SHA,FROZEN_ADAPTER_SHA
from h136.train_gpu import BASE_MODEL,BASE_REV
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h133.paired_stats import exact_discordant_p

ARMS=("base_pretrained","h136_64step_adapter")

def verify_raw(source,report,raw):
    if (report.get("schema")!=FORMAT or
        report.get("status")!="H136_REAL_ONCE_ONLY_SEALED24_NEURAL_COMPLETE" or
        report.get("frozen_original_source_manifest_sha256")!=SEALED_MANIFEST_SHA or
        report.get("frozen_adapter_sha256")!=FROZEN_ADAPTER_SHA or
        report.get("base_model")!=BASE_MODEL or report.get("base_revision")!=BASE_REV or
        report.get("sealed_new_scene_count")!=24 or
        report.get("neural_inference_calls")!=96 or
        report.get("actual_optimizer_steps")!=0 or
        report.get("train_or_dev_images_attached") is not False or
        report.get("paid_compute_used") is not False):
        raise ValueError("H136 once-only heldout frozen GPU evidence invalid")
    if not isinstance(raw,list) or len(raw)!=96:
        raise ValueError("H136 owner requires exact 96 raw neural outputs")
    indexed={}
    i=0
    for arm in ARMS:
        for scene in source["cases"]:
            for role in ("start","end"):
                obs=raw[i]
                i+=1
                if (obs.get("arm")!=arm or obs.get("case_id")!=scene["id"] or
                    obs.get("image_sha256")!=scene["image_sha256"] or obs.get("role")!=role):
                    raise ValueError("H136 heldout model case/arm/role mismatch")
                parsed=native_point(obs.get("raw"),WIDTH,HEIGHT)
                if obs.get("point")!=parsed:
                    raise ValueError("H136 native model text/point integrity failed")
                box=scene[role+"_bbox"]
                within=parsed is not None and (
                    box["x"]<=parsed[0]<=box["x"]+box["width"] and
                    box["y"]<=parsed[1]<=box["y"]+box["height"])
                if obs.get("hit") is not bool(within):
                    raise ValueError("H136 claimed neural bounding-box hit not independently reproducible")
                indexed[(arm,scene["id"],role)]=parsed
    return indexed

def replay(source_path,output_dir,evidence_path):
    source_path,output_dir,evidence_path=map(Path,(source_path,output_dir,evidence_path))
    source=check(source_path,"sealed")
    if sha((source_path/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H136 preregistered sealed screenshot source changed")
    report=json.loads((output_dir/"private-sealed24-evidence.json").read_text("utf8"))
    raw=json.loads((output_dir/"private-sealed24-raw.json").read_text("utf8"))
    points=verify_raw(source,report,raw)
    from playwright.sync_api import sync_playwright
    fixture=Path(__file__).with_name("fixture.html").resolve()
    tasks={}
    actual_drag_count=0
    with sync_playwright() as playwright:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:options["executable_path"]=binary
        browser=playwright.chromium.launch(**options)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route(re.compile(r"^https?://"),lambda route:route.abort())
            page=ctx.new_page()
            for arm in ARMS:
                done=[]
                for scene in source["cases"]:
                    seed=scene["id"].split("-")[-1]
                    page.goto(fixture.as_uri()+"?id="+seed,wait_until="load")
                    page.locator('[data-role="handle"]').wait_for()
                    orig=page.screenshot(type="png")
                    if sha(orig)!=scene["image_sha256"]:
                        raise ValueError("H136 Chromium heldout original screenshot SHA drift")
                    start=points[(arm,scene["id"],"start")]
                    end=points[(arm,scene["id"],"end")]
                    if start is None or end is None:
                        task_ok=False
                    else:
                        actual_drag_count+=1
                        page.mouse.move(*start)
                        page.mouse.down()
                        page.mouse.move(*end,steps=12)
                        page.mouse.up()
                        task_ok=page.locator("#canvas").get_attribute("data-success")=="true"
                    done.append({"scene":scene["id"],"success":task_ok})
                tasks[arm]=done
            ctx.close()
        finally:browser.close()
    base={x["scene"]:x["success"] for x in tasks[ARMS[0]]}
    trained={x["scene"]:x["success"] for x in tasks[ARMS[1]]}
    trained_only=sum(not base[k] and trained[k] for k in base)
    base_only=sum(base[k] and not trained[k] for k in base)
    evidence={
        "status":"PASS_H136_ONCE_ONLY_SEALED24_REAL_BROWSER_REPLAY",
        "frozen_adapter_sha256":FROZEN_ADAPTER_SHA,
        "sealed_source_sha256":SEALED_MANIFEST_SHA,
        "evaluated_model_scene_pairs":48,
        "actual_owner_gui_mouse_actions":actual_drag_count,
        "actual_neural_calls":96,"actual_optimizer_steps":0,
        "new_original_heldout_scenes":24,
        "pretrained_gui_task_successes":sum(base.values()),
        "trained_adapter_gui_task_successes":sum(trained.values()),
        "paired_trained_only":trained_only,
        "paired_pretrained_only":base_only,
        "exact_p_two_sided":exact_discordant_p(trained_only,base_only),
        "flagship_Hala_35B_model_improved_proven":False,
        "generalizable_model_skill_proven":False,
        "real_application_benchmark_proven":False,
    }
    if evidence_path.exists():
        raise ValueError("H136 owner sealed report create-only")
    evidence_path.parent.mkdir(parents=True,exist_ok=True)
    evidence_path.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n","utf8")
    return evidence

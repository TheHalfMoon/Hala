"""H151 ONE-SHOT frozen model-driven ORIGINAL shifted-layout mouse test.

Source pixel replay must pass BEFORE consuming private predictions. The
actuator receives ONLY the H142 LoRA predicted START and visible heading
points. Owner DOM gold used exclusively afterward for scoring. The source
is unmodified MIT React app with owner browser-only layout CSS changes.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h150.visual_region_controller import visually_center_drop
from h141.interactive_source import verify_owner_original
from h151.freeze_eval import audit,FOLDER
from h151.gpu_probe import (
    SCHEMA as PREDICTIONS_SCHEMA,IMAGE_MANIFEST_SHA,
    OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA
)
from h151.collect_shifted_source import source_case
from h151.replay import replay,reconstruct

PREDICTIONS_FIELDS={
    "id","app","language","destination_status","layout",
    "start_point","start_output_valid",
    "visible_column_heading_point","visible_column_heading_output_valid"
}
LOCK="H151_SHIFTED_LAYOUT_ONCE_MODEL_ACTION_CONSUMED.json"
RESULT="H151_SHIFTED_LAYOUT_ONCE_REAL_BROWSER_OUTCOMES.json"

def valid_point(p):
    return (isinstance(p,list) and len(p)==2 and
            all(type(v) in (int,float) and math.isfinite(v) for v in p)
            and 0<=p[0]<WIDTH and 0<=p[1]<HEIGHT)

def predictions(source_file,owner_root):
    data=json.loads(Path(source_file).read_text("utf8"))
    allowed=audit(owner_root)
    if (allowed["manifest_sha256"]!=OWNER_SOURCE_MANIFEST_SHA or
        data.get("schema")!=PREDICTIONS_SCHEMA or
        data.get("status")!="COMPLETE_REAL_MODEL_SHIFTED_LAYOUT_32_INFERENCE_CALLS" or
        data.get("original_new_source_manifest_sha256")!=OWNER_SOURCE_MANIFEST_SHA or
        data.get("input_manifest_sha256")!=IMAGE_MANIFEST_SHA or
        data.get("model_checkpoint_sha256")!=ADAPTER_MODEL_SHA or
        data.get("gpu_calls")!=32 or
        data.get("optimizer_steps")!=0 or
        data.get("owner_gold_boxes_available_to_model") is not False or
        data.get("h140_sealed_blind_calls")!=0 or
        data.get("flagship_Hala_35B_updated") is not False):
        raise ValueError("H151 private original shifted neural output not proven")
    cohort=json.loads((Path(owner_root)/"private-h151"/FOLDER/
                       "manifest.json").read_text("utf8"))
    expected={(r["id"],r["layout"],r["language"],r["destination_status"])
              for r in cohort["cases"]}
    if not isinstance(data.get("predictions"),list) or len(data["predictions"])!=16:
        raise ValueError("H151 new shifted original model output count must equal 16")
    result={}
    for r in data["predictions"]:
        if not isinstance(r,dict) or set(r)!=PREDICTIONS_FIELDS:
            raise ValueError("H151 private model inference output must not contain owner oracle")
        key=(r["id"],r["layout"],r["language"],r["destination_status"])
        if key not in expected or key in result:
            raise ValueError("H151 original model output task IDs do not match owner source")
        for role in ("start","visible_column_heading"):
            p=r[role+"_point"]
            if type(r[role+"_output_valid"]) is not bool or (
                r[role+"_output_valid"]!=valid_point(p)):
                raise ValueError("H151 model output point validity conflict")
        result[key]=r
    if set(result)!=expected:
        raise ValueError("H151 missing model predictions")
    return result,cohort

def run(owner_root,predictions_file):
    from playwright.sync_api import sync_playwright
    owner=Path(owner_root)
    # Fail CLOSED: first prove the 16 original images are identically
    # reproducible, and the 32 neural outputs admit no owner gold.
    source_probe=replay(owner)
    prediction_rows,source=predictions(predictions_file,owner)
    if source_probe["cases"]!=16:
        raise ValueError("H151 new original pixel replay insufficient")
    locked=owner/"private-h151"/LOCK
    with locked.open("x",encoding="utf8") as fh:
        fh.write(json.dumps({
            "schema":"hala.h151.one-time-original-model-browser-lock.v1",
            "status":"ORIGINAL_NEURAL_MODEL_TEST_CONSUMED",
            "owner_manifest_sha":OWNER_SOURCE_MANIFEST_SHA,
            "adapter_sha":ADAPTER_MODEL_SHA,
            "h140_blind_calls":0},indent=2)+"\n")
    source_dir=verify_owner_original(owner/"private-h140"/"upstream"/"react-native-board")
    results=[];counts=defaultdict(Counter)
    with sync_playwright() as pw:
        kwargs={"headless":True}
        binary=_chromium_binary()
        if binary:kwargs["executable_path"]=binary
        browser=pw.chromium.launch(**kwargs,timeout=20000)
        try:
            with serve(source_dir) as url:
                for row in source["cases"]:
                    task=source_case(row["attempt_index"])
                    key=(row["id"],row["layout"],row["language"],row["destination_status"])
                    pred=prediction_rows[key]
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(task["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        reconstruct(page,row)
                        start=pred["start_point"]
                        heading=pred["visible_column_heading_point"]
                        def hits(point,rect):
                            return bool(valid_point(point) and
                                rect["x"]<=point[0]<=rect["x"]+rect["width"] and
                                rect["y"]<=point[1]<=rect["y"]+rect["height"])
                        start_hit=hits(start,row["start_bbox"])
                        heading_hit=hits(
                            heading,row["visible_destination_column_heading_bbox"])
                        changed=persisted=False
                        visual_found=False
                        signal_retained=False
                        if valid_point(start) and valid_point(heading):
                            # The native mouse controller sees ONLY
                            # neural predicted START and heading points,
                            # plus live screenshots; never true DOM boxes.
                            try:
                                feedback=visually_center_drop(page,start,heading)
                                visual_found=feedback["visible_region_found"]
                                signal_retained=feedback["retained_visual_change_at_release"]
                                page.wait_for_timeout(120)
                                now=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                changed=any(x["task"]==task["title"] and
                                            x["status"]==task["destination_status"]
                                            for x in now)
                                page.reload(wait_until="commit",timeout=18000)
                                later=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                persisted=changed and any(
                                    x["task"]==task["title"] and
                                    x["status"]==task["destination_status"]
                                    for x in later)
                            except (ValueError,TimeoutError):
                                # Never replace model outputs with oracle.
                                pass
                        outcome={"id":row["id"],
                                 "language":row["language"],
                                 "destination_status":row["destination_status"],
                                 "layout":row["layout"],
                                 "source_model_point_hits":start_hit,
                                 "heading_model_point_hits":heading_hit,
                                 "both_model_points_hit":start_hit and heading_hit,
                                 "visual_drop_region_detected":visual_found,
                                 "visual_drop_signal_retained":signal_retained,
                                 "true_original_task_changed":bool(changed),
                                 "true_original_task_persisted_after_reload":bool(persisted),
                                 "original_screenshot_sha_replayed":True}
                        results.append(outcome)
                        for name in ("all",row["layout"],row["language"],
                                     row["destination_status"]):
                            counts[name]["cases"]+=1
                            for metric in ("source_model_point_hits",
                                           "heading_model_point_hits",
                                           "both_model_points_hit",
                                           "visual_drop_region_detected",
                                           "visual_drop_signal_retained",
                                           "true_original_task_changed",
                                           "true_original_task_persisted_after_reload"):
                                counts[name][metric]+=int(outcome[metric])
                        if len(results)%4==0:
                            print("H151_REAL_OWNER_NATIVE_MODEL_DRAGS",
                                  len(results),16,flush=True)
                    finally:ctx.close()
        finally:browser.close()
    if len(results)!=16:raise ValueError("H151 model-driven original cohort incomplete")
    report={
        "schema":"hala.h151.one-time-independent-shifted-layout-model-task-evaluation.v1",
        "status":"H151_FROZEN_SHIFTED_MODEL_ACTION_RESULT_NOT_RELEASE",
        "new_source_cases":16,
        "model_neural_calls":32,
        "real_h142_training_steps_on_this_source":0,
        "model_points_not_owner_gold_drive_mouse":True,
        "screenshot_only_drop_center_controller":True,
        "base_2B_adapter_sha":ADAPTER_MODEL_SHA,
        "source_manifest_sha":OWNER_SOURCE_MANIFEST_SHA,
        "h140_blind_neural_calls":0,
        "main_hala_35B_updated":False,
        "release_qualified":False,
        "summary":{k:dict(v) for k,v in counts.items()},
    }
    with (owner/"private-h151"/RESULT).open("x",encoding="utf8") as fp:
        fp.write(json.dumps({"report":report,"private_outcomes":results},
                            indent=2,sort_keys=True)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--predictions",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.predictions),indent=2,sort_keys=True))

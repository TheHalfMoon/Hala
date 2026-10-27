"""H153 ONE-SHOT frozen model-driven ORIGINAL shifted-layout mouse test.

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
from h152.horizontal_visual_search import search_horizontal_visible_release
from h141.interactive_source import verify_owner_original
from h153.freeze_eval import audit,FOLDER
from h153.gpu_probe import (
    SCHEMA as PREDICTIONS_SCHEMA,IMAGE_MANIFEST_SHA,
    OWNER_SOURCE_MANIFEST_SHA,ADAPTER_MODEL_SHA
)
from h153.collect_shifted_source import source_case
from h153.replay import replay,reconstruct

PREDICTIONS_FIELDS={
    "id","app","language","destination_status","layout",
    "start_point","start_output_valid",
    "visible_column_heading_point","visible_column_heading_output_valid"
}
LOCK="H153_SHIFTED_LAYOUT_ONCE_MODEL_ACTION_CONSUMED.json"
RESULT="H153_SHIFTED_LAYOUT_ONCE_REAL_BROWSER_OUTCOMES.json"

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
        raise ValueError("H153 private original shifted neural output not proven")
    cohort=json.loads((Path(owner_root)/"private-h153"/FOLDER/
                       "manifest.json").read_text("utf8"))
    expected={(r["id"],r["layout"],r["language"],r["destination_status"])
              for r in cohort["cases"]}
    if not isinstance(data.get("predictions"),list) or len(data["predictions"])!=16:
        raise ValueError("H153 new shifted original model output count must equal 16")
    result={}
    for r in data["predictions"]:
        if not isinstance(r,dict) or set(r)!=PREDICTIONS_FIELDS:
            raise ValueError("H153 private model inference output must not contain owner oracle")
        key=(r["id"],r["layout"],r["language"],r["destination_status"])
        if key not in expected or key in result:
            raise ValueError("H153 original model output task IDs do not match owner source")
        for role in ("start","visible_column_heading"):
            p=r[role+"_point"]
            if type(r[role+"_output_valid"]) is not bool or (
                r[role+"_output_valid"]!=valid_point(p)):
                raise ValueError("H153 model output point validity conflict")
        result[key]=r
    if set(result)!=expected:
        raise ValueError("H153 missing model predictions")
    return result,cohort

def run(owner_root,predictions_file):
    from playwright.sync_api import sync_playwright
    owner=Path(owner_root)
    # Admit independent exact screenshot replay, full 32 model predictions,
    # and owner SHA before consuming this one-shot paired cohort.
    prior=replay(owner)
    predicted,cohort=predictions(predictions_file,owner)
    if prior.get("cases")!=16:
        raise ValueError("H153 original model screenshot replay incomplete")
    lock=owner/"private-h153"/LOCK
    with lock.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "schema":"hala.h153.one-time-neural-controller-paired-evaluation-lock.v1",
            "status":"H153_NEURAL_MODEL_EVALUATION_CONSUMED",
            "source_manifest_sha256":OWNER_SOURCE_MANIFEST_SHA,
            "adapter_model_sha256":ADAPTER_MODEL_SHA,
            "sealed_h140_blind_calls":0},indent=2)+"\\n")
    original=verify_owner_original(
        owner/"private-h140"/"upstream"/"react-native-board")
    outcomes=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as url:
                for row in cohort["cases"]:
                    key=(row["id"],row["layout"],row["language"],row["destination_status"])
                    model=predicted[key]
                    task=source_case(row["attempt_index"])
                    start=model["start_point"]
                    goal=model["visible_column_heading_point"]
                    def hit(p,box):
                        return bool(valid_point(p) and
                                    box["x"]<=p[0]<=box["x"]+box["width"] and
                                    box["y"]<=p[1]<=box["y"]+box["height"])
                    source_hit=hit(start,row["start_bbox"])
                    header_hit=hit(goal,row["visible_destination_column_heading_bbox"])
                    record={
                        "id":row["id"],"language":row["language"],
                        "destination_status":row["destination_status"],
                        "layout":row["layout"],
                        "source_model_point_hits":source_hit,
                        "visible_heading_model_point_hits":header_hit,
                        "both_point_hits":source_hit and header_hit,
                    }
                    for arm in ("H150","H152"):
                        ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                device_scale_factor=1,service_workers="block")
                        ctx.add_init_script(
                            "if(localStorage.getItem('tasks')===null){"+
                            "localStorage.setItem('tasks',"+
                            json.dumps(json.dumps(task["rows"]))+");}")
                        ctx.route("https://**/*",lambda route:route.abort())
                        moved=saved=False
                        found=retained=False
                        try:
                            page=ctx.new_page()
                            page.goto(url,wait_until="commit",timeout=18000)
                            reconstruct(page,row)
                            if valid_point(start) and valid_point(goal):
                                try:
                                    feedback=(visually_center_drop(page,start,goal)
                                              if arm=="H150" else
                                              search_horizontal_visible_release(page,start,goal))
                                    found=feedback["visible_region_found"]
                                    retained=(feedback["retained_visual_change_at_release"]
                                              if arm=="H150" else
                                              feedback["visual_signal_retained"])
                                    page.wait_for_timeout(120)
                                    now=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                    moved=any(
                                        x["task"]==task["title"] and
                                        x["status"]==task["destination_status"]
                                        for x in now)
                                    page.reload(wait_until="commit",timeout=18000)
                                    later=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                    saved=moved and any(
                                        x["task"]==task["title"] and
                                        x["status"]==task["destination_status"]
                                        for x in later)
                                except (ValueError,TimeoutError):
                                    pass
                        finally:ctx.close()
                        record[arm+"_visible_region_found"]=bool(found)
                        record[arm+"_visual_signal_retained"]=bool(retained)
                        record[arm+"_persisted"]=bool(saved)
                    outcomes.append(record)
                    if len(outcomes)%4==0:
                        print("H153_NEW_TRUE_NEURAL_CONTROLLER_PAIRED_ACTIONS",
                              len(outcomes),16,flush=True)
        finally:browser.close()
    if len(outcomes)!=16:
        raise ValueError("H153 native paired evaluation did not complete")
    report={
        "schema":"hala.h153.true-neural-one-shot-controller-paired-a-b.v1",
        "status":"H153_NEW_FROZEN_MODEL_ACTUAL_PAIRED_GUI_OUTCOMES",
        "new_cases":16,"model_calls":32,
        "genuine_native_mouse_drags":32,
        "model_source_hits":sum(r["source_model_point_hits"] for r in outcomes),
        "model_heading_hits":sum(r["visible_heading_model_point_hits"] for r in outcomes),
        "model_both_hits":sum(r["both_point_hits"] for r in outcomes),
        "old_H150_persisted":sum(r["H150_persisted"] for r in outcomes),
        "new_H152_persisted":sum(r["H152_persisted"] for r in outcomes),
        "H152_only_success":sum(r["H152_persisted"] and not r["H150_persisted"] for r in outcomes),
        "H150_only_success":sum(r["H150_persisted"] and not r["H152_persisted"] for r in outcomes),
        "H152_feedback_stable":sum(r["H152_visual_signal_retained"] for r in outcomes),
        "H150_feedback_stable":sum(r["H150_visual_signal_retained"] for r in outcomes),
        "source_sha":OWNER_SOURCE_MANIFEST_SHA,
        "frozen_external_2B_adapter_sha":ADAPTER_MODEL_SHA,
        "model_and_controller_saw_hidden_oracle_dom":False,
        "h140_sealed_blind_calls":0,"optimizer_steps":0,
        "Hala_flagship_35B_updated":False,
        "release_qualified":False,
    }
    target=owner/"private-h153"/"H153_TRUE_NEURAL_PAIRED_CONTROLLER_RESULT.json"
    with target.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":report,"private_original_episodes":outcomes},
                           sort_keys=True,indent=2)+"\\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--predictions",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.predictions),indent=2,sort_keys=True))

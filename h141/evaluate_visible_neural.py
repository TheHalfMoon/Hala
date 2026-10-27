"""H141 ONE-SHOT original React visible-target NEURAL controller evaluation.

Predictor receives no original DOM gold. The screenshot is replayed EXACTLY
on owner Windows and the actuator gets ONLY model points, never DOM bbox.
Owner DOM truth is read exclusively AFTER predictions for scoring. This is
a new H141 evaluation cohort, not H140's sealed independent OOD BLIND.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.qualification import digest
from h141.interactive_source import original_case,verify_owner_original
from h141.coarse_column_action import move_card_by_coarse_visible_goal
from h141.gpu_visible_probe import (
    SCHEMA as NEURAL_SCHEMA,STATUS,EVAL_INPUT_SHA,EVAL_MANIFEST_SHA,
    V2_ADAPTER_SHA,SOURCE_SHA
)
from h141.qualify_neural_probe import PRIVATE_FOLDER,EVAL_SCHEMA,audit

LOCK_FILENAME="H141_ONCE_MODEL_VISIBLE_EVALUATION_LOCK.json"
RESULT_FILENAME="H141_REAL_NEURAL_VISIBLE_BROWSER_RESULT.json"
FIELDS={"id","app","language","start_point","start_output_valid",
        "visible_column_heading_point","visible_column_heading_output_valid"}

def admissible_point(p):
    return (isinstance(p,list) and len(p)==2 and
            all(type(x) in (int,float) and math.isfinite(x) for x in p) and
            0<=p[0]<WIDTH and 0<=p[1]<HEIGHT)

def load_predictions(predictions,owner_root):
    doc=json.loads(Path(predictions).read_text("utf8"))
    original=json.loads((Path(owner_root)/"private-h141"/PRIVATE_FOLDER/
                         "manifest.json").read_text("utf8"))
    certified=audit(owner_root)
    if (original.get("schema")!=EVAL_SCHEMA or
        certified.get("source_manifest_sha256")!=
            "20a472485cd04f3b6284c0a8d8e7ab01b5ff1f09154d8b5ab4479a313da8ad3b" or
        doc.get("schema")!=NEURAL_SCHEMA or
        doc.get("status")!=STATUS or
        doc.get("original_source_role")!="new_h141_neural_probe_not_h140_blind" or
        doc.get("original_source_git_sha")!=SOURCE_SHA or
        doc.get("input_zip_sha256")!=EVAL_INPUT_SHA or
        doc.get("input_manifest_sha256")!=EVAL_MANIFEST_SHA or
        doc.get("adapter_sha256")!=V2_ADAPTER_SHA or
        doc.get("gold_boxes_seen_by_model") is not False or
        doc.get("training_eligible") is not False or
        doc.get("real_optimizer_steps")!=0 or
        doc.get("sealed_h140_blind_calls")!=0 or
        doc.get("model_calls")!=32):
        raise ValueError("H141 actual model probes not original, certified or gold-free")
    expected={(r["id"],r["language"],r["app"]) for r in original["cases"]}
    rows=doc.get("predictions")
    if not isinstance(rows,list) or len(rows)!=16:
        raise ValueError("H141 exactly 16 original screenshot model predictions")
    found={}
    for row in rows:
        if not isinstance(row,dict) or set(row)!=FIELDS:
            raise ValueError("H141 no oracle/cursor data permitted in neural results")
        key=(row["id"],row["language"],row["app"])
        if key not in expected or key in found:
            raise ValueError("H141 predicted original source identity mismatch")
        for role in ("start","visible_column_heading"):
            p=row[role+"_point"]
            valid=row[role+"_output_valid"]
            if type(valid) is not bool or valid!=admissible_point(p) and p is not None:
                raise ValueError("H141 inferred original point validity mismatch")
            if p is None and valid:
                raise ValueError("H141 missing output cannot report valid point")
        found[key]=row
    if set(found)!=expected:
        raise ValueError("H141 missing one or more original verified GUI cases")
    return found,original

def run(owner_root,predictions):
    from playwright.sync_api import sync_playwright
    root=Path(owner_root)
    found,doc=load_predictions(predictions,root)
    folder=root/"private-h141"/PRIVATE_FOLDER
    lock=root/"private-h141"/LOCK_FILENAME
    with lock.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "schema":"hala.h141.once-visible-original-model-action-lock.v1",
            "status":"CONSUMED_NO_AUTOMATIC_RETRY",
            "owner_source_sha256":"20a472485cd04f3b6284c0a8d8e7ab01b5ff1f09154d8b5ab4479a313da8ad3b",
            "sealed_h140_blind_calls":0,
        },indent=2)+"\n")
    original_built=verify_owner_original(root/"private-h140"/"upstream"/"react-native-board")
    results=[];metric=defaultdict(Counter)
    with sync_playwright() as pw:
        config={"headless":True}
        browser_exe=_chromium_binary()
        if browser_exe:config["executable_path"]=browser_exe
        browser=pw.chromium.launch(**config,timeout=20000)
        try:
            with serve(original_built) as url:
                for row in doc["cases"]:
                    key=(row["id"],row["language"],row["app"])
                    prediction=found[key]
                    seed=original_case(row["attempt_index"])
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(seed["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        observed=digest(page.screenshot(type="png"))
                        if observed!=row["image_sha256"]:
                            raise ValueError("H141 original model-observed source screenshot not SHA-identical")
                        src=prediction["start_point"]
                        dst=prediction["visible_column_heading_point"]
                        def hit(p,box):
                            return bool(admissible_point(p) and
                                box["x"]<=p[0]<=box["x"]+box["width"] and
                                box["y"]<=p[1]<=box["y"]+box["height"])
                        start_hit=hit(src,row["start_bbox"])
                        heading_hit=hit(dst,row["visible_destination_column_heading_bbox"])
                        changed=False;persisted=False
                        if admissible_point(src) and admissible_point(dst):
                            # This temporary heading box is created from the
                            # MODEL point only — NEVER the DOM ground truth.
                            model_heading={"x":dst[0]-60,"y":dst[1]-15.5,
                                           "width":120,"height":31}
                            try:
                                move_card_by_coarse_visible_goal(page,src,model_heading)
                                page.wait_for_timeout(120)
                                now=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                changed=sum(x["task"]==seed["title"] and
                                            x["status"]=="doing" for x in now)==1
                                page.reload(wait_until="commit",timeout=18000)
                                after=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                persisted=sum(x["task"]==seed["title"] and
                                              x["status"]=="doing" for x in after)==1
                            except ValueError:
                                # Non-actionable model point means failed
                                # task; never substitute oracle coordinates.
                                pass
                        record={
                            "id":row["id"],"app":row["app"],"language":row["language"],
                            "start_point_hits_visible_source":start_hit,
                            "heading_point_hits_visible_label":heading_hit,
                            "both_points_hit":bool(start_hit and heading_hit),
                            "original_browser_task_status_changed":changed,
                            "original_browser_task_persisted":bool(changed and persisted),
                            "exact_original_input_sha_replayed":True,
                        }
                        results.append(record)
                        for category in ("all",row["language"]):
                            for metric_name in ("start_point_hits_visible_source",
                                                "heading_point_hits_visible_label",
                                                "both_points_hit",
                                                "original_browser_task_status_changed",
                                                "original_browser_task_persisted"):
                                metric[category][metric_name]+=int(record[metric_name])
                            metric[category]["cases"]+=1
                        if len(results)%4==0:
                            print("H141_REAL_MODEL_PREDICTED_VISIBLE_COLUMN_MOUSE_ACTION",
                                  len(results),16,flush=True)
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if len(results)!=16:
        raise ValueError("H141 real original screenshot cohort incomplete")
    report={
        "schema":"hala.h141.once-real-original-neural-visible-column-e2e.v1",
        "status":"ORIGINAL_SOURCE_MODEL_PREDICTED_ACTION_EVALUATED_NOT_BLIND",
        "cases":16,"app_codebases":1,
        "model_guessed_source_and_visible_heading":True,
        "original_source_reference_dom_proposals_used_by_controller":False,
        "model_trained_on_eval":False,
        "training_optimizer_steps":0,
        "sealed_h140_blind_calls":0,
        "source_model":"original_h140_v2_2b_adapter",
        "per_language":{k:dict(v) for k,v in metric.items()},
        "flagship_Hala_35B_updated":False,
        "release_qualified":False,
    }
    result_file=root/"private-h141"/RESULT_FILENAME
    with result_file.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":report,
                            "owner_private_case_outcomes":results},sort_keys=True,indent=2)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--predictions",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner_root,args.predictions),sort_keys=True,indent=2))

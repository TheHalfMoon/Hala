"""H133 preregistered independent paired real Chromium GUI evaluation.

Read ONLY original private screenshot/response when choosing actions; actual
DOM handle geometry is inspected AFTER frozen model point exists, for scoring.
"""
import json
import statistics
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT
from h133.browser_env import CASES,UnseenBrowserEnv
from h133.source import validate,sha
from h133.neural import FORMAT
from h131.contract import parse_showui
from h132.grounder_contract import MODEL_ID,MODEL_SHA,native_point
from h125.model_contracts import MODELS
from h131.owner_diagnostics import distance_to_box

def verify_model(file,manifest,arm):
    data=json.loads(Path(file).read_text("utf8"))
    if arm=="showui":
        model_id,model_sha=MODELS["showui_2b"].model_id,MODELS["showui_2b"].revision_sha
        parser=parse_showui
    elif arm=="specialist":
        model_id,model_sha=MODEL_ID,MODEL_SHA
        parser=native_point
    else:
        raise ValueError("H133 exactly two preregistered model arms")
    if (data.get("schema")!=FORMAT or data.get("model_arm")!=arm or
        data.get("model_id")!=model_id or data.get("model_sha")!=model_sha or
        data.get("source_manifest_sha256")!=sha(json.dumps(manifest,
            sort_keys=True,separators=(",",":")).encode("utf8")) or
        data.get("real_neural_calls")!=16 or
        data.get("training_steps")!=0 or
        data.get("oracle_coordinates_in_model_input") is not False):
        raise ValueError("H133 frozen model/source evidence mismatch")
    rows=data.get("cases")
    if not isinstance(rows,list) or len(rows)!=8:
        raise ValueError("H133 exactly eight real neural paired cases")
    by_task={}
    for case,record in zip(manifest["cases"],rows):
        if record.get("task")!=case["task"] or record.get("image_sha256")!=case["image_sha256"]:
            raise ValueError("H133 pretrained model result scene identity mismatch")
        for role in ("start","end"):
            raw=record.get("raw_"+role)
            parsed=parser(raw,WIDTH,HEIGHT)
            result=record.get(role)
            if parsed is None:
                if result is not None:
                    raise ValueError("H133 unparseable model response must not be replaced")
            elif list(parsed)!=result or type(result) is not list:
                raise ValueError("H133 model coordinate does not match real raw response")
        by_task[case["task"]]=record
    return by_task

def evaluate(source_dir,showui_file,specialist_file,output_file):
    manifest=validate(source_dir)
    arm_data={arm:verify_model(path,manifest,arm) for arm,path in
              (("showui",showui_file),("specialist",specialist_file))}
    totals={}
    for arm in ("showui","specialist"):
        details=[]
        for case in manifest["cases"]:
            task=case["task"]
            point=arm_data[arm][task]
            with UnseenBrowserEnv(task) as env:
                old=env.observe()
                if old.image_sha256!=case["image_sha256"]:
                    raise ValueError("H133 local Chromium changed from frozen source")
                # After-inference ground truth: never used to select action.
                handle=env._page.locator('[data-role="handle"]').bounding_box()
                target=env._page.locator("#target").bounding_box()
                if handle is None or target is None:
                    raise ValueError("H133 synthetic browser QA controls missing")
                start_err=(distance_to_box(point["start"],handle)
                            if point["start"] is not None else None)
                end_err=(distance_to_box(point["end"],target)
                          if point["end"] is not None else None)
                valid=point["start"] is not None and point["end"] is not None
                if valid:
                    feedback=env.execute_drag(point["start"],point["end"])
                    success=feedback.success
                    change=old.image_sha256!=feedback.observation.image_sha256
                else:
                    success=change=False
                details.append({
                    "case":task,"valid_native_pair":valid,
                    "start_hits_handle":start_err==0 if start_err is not None else False,
                    "start_distance_px":round(start_err,2) if start_err is not None else None,
                    "end_distance_px":round(end_err,2) if end_err is not None else None,
                    "actual_gui_task_success":success,"visible_state_changed":change})
        errors=[x["start_distance_px"] for x in details if x["start_distance_px"] is not None]
        totals[arm]={
            "count":len(details),"native_valid_pairs":sum(x["valid_native_pair"] for x in details),
            "handle_hits":sum(x["start_hits_handle"] for x in details),
            "task_successes":sum(x["actual_gui_task_success"] for x in details),
            "visual_changes":sum(x["visible_state_changed"] for x in details),
            "median_start_error_px":statistics.median(errors) if errors else None,
            "per_case":details}
    outcome={
        "schema":"hala.h133.owner-independent-unseen8-chromium-comparison.v1",
        "status":"H133_TWO_FROZEN_MODELS_REAL_BROWSER_OUTCOMES_SCORED",
        "source_scene_count":8,
        "models":totals,
        "scene_selection_before_model_inference":True,
        "zero_model_training":True,
        "four_initial_H131_scenes_excluded":True,
        "original_H71_training_or_EVAL_TEST_used":False,
        "model_skill_qualified":False,
        "limited_power_warning":"8 synthetic browser tasks; not a qualified real-world benchmark",
    }
    output_file=Path(output_file)
    if output_file.exists():
        raise ValueError("H133 research outcome evidence create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(outcome,sort_keys=True,indent=2)+"\n","utf8")
    return outcome

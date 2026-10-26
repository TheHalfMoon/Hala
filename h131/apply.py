"""Execute true pretrained neural GUI coordinates in actual local Chromium.

No DOM/ground-truth coordinates used for mouse actions. Two-stage second
attempt replays the first stage deterministically in a new isolated browser.
"""
import json
from pathlib import Path
from h130.browser_env import BrowserDragEnv, TASKS, WIDTH, HEIGHT
from h131.contract import (SCHEMA, sha, verify_manifest,
                           verify_model_predictions)

def _read(path):
    return json.loads(Path(path).read_text("utf8"))

def _store(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n","utf8")

def apply_initial(input_dir, native_file, private_output_dir):
    input_dir=Path(input_dir);private_output_dir=Path(private_output_dir)
    if private_output_dir.exists():
        raise ValueError("H131 private output must be fresh")
    manifest=verify_manifest(input_dir,expect_phase="initial")
    predictions=verify_model_predictions(_read(native_file),manifest)
    private_output_dir.mkdir(parents=True)
    phase_dir=private_output_dir/"feedback"
    phase_dir.mkdir()
    feedback_rows=[];evidence=[]
    for case in manifest["cases"]:
        task=case["task"]
        pred=predictions[task]
        with BrowserDragEnv(task) as env:
            observation=env.observe()
            if observation.image_sha256!=case["image_sha256"]:
                raise ValueError("H131 source scene changed since original neural screenshot")
            outcome=env.execute_drag(pred["start"],pred["end"])
            (phase_dir/(task+".png")).write_bytes(outcome.observation.screenshot_png)
            feedback_rows.append({"task":task,"image":task+".png",
                "image_sha256":outcome.observation.image_sha256,
                "instruction":outcome.observation.instruction,
                "previous_action_succeeded":outcome.success})
            evidence.append({"task":task,"first_action_applied":outcome.action_applied,
                "first_success":outcome.success,
                "actual_browser_action_count":1,
                "before_sha256":observation.image_sha256,
                "after_sha256":outcome.observation.image_sha256})
    feedback_manifest={"schema":SCHEMA,"phase":"feedback",
        "width":WIDTH,"height":HEIGHT,"cases":feedback_rows,
        "real_browser_snapshots":True,"oracle_browser_geometry_exposed":False,
        "real_model_inference_used":True}
    _store(phase_dir/"manifest.json",feedback_manifest)
    result={"status":"PASS_H131_FIRST_REAL_NEURAL_BROWSER_ACTIONS",
            "source_manifest_sha256":sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8")),
            "feedback_manifest_sha256":sha(json.dumps(feedback_manifest,sort_keys=True,separators=(",",":")).encode("utf8")),
            "model_generated_actions":4,
            "browser_successes":sum(x["first_success"] for x in evidence),
            "browser_attempts":4,"phase":"initial","model_skill_qualified":False,
            "per_task":evidence}
    _store(private_output_dir/"private-initial-outcomes.json",result)
    return result

def apply_feedback(initial_dir, initial_model_file, feedback_dir,
                   feedback_model_file, private_output_path):
    initial=verify_manifest(initial_dir,expect_phase="initial")
    subsequent=verify_manifest(feedback_dir,expect_phase="feedback")
    initial_points=verify_model_predictions(_read(initial_model_file),initial)
    feedback_points=verify_model_predictions(_read(feedback_model_file),subsequent)
    tasks_initial={x["task"]:x for x in initial["cases"]}
    tasks_feedback={x["task"]:x for x in subsequent["cases"]}
    rows=[]
    for task in TASKS:
        first=initial_points[task];second=feedback_points[task]
        with BrowserDragEnv(task) as env:
            first_scene=env.observe()
            if first_scene.image_sha256!=tasks_initial[task]["image_sha256"]:
                raise ValueError("H131 deterministic first screenshot mismatch")
            after_first=env.execute_drag(first["start"],first["end"])
            if (after_first.observation.image_sha256!=tasks_feedback[task]["image_sha256"] or
                after_first.success!=tasks_feedback[task].get("previous_action_succeeded")):
                raise ValueError("H131 failure to replay exact original browser feedback state")
            after_second=None
            if not after_first.success:
                after_second=env.execute_drag(second["start"],second["end"])
            rows.append({"task":task,"first_success":after_first.success,
                         "second_action_executed":after_second is not None,
                         "second_success":after_second.success if after_second else None,
                         "final_success":after_first.success or bool(after_second and after_second.success)})
    results={"schema":"hala.h131.private-real-neural-browser-feedback-results.v1",
             "status":"H131_REAL_NEURAL_TWO_STAGE_BROWSER_FEEDBACK_SCORED",
             "task_count":len(rows),"first_pass_successes":sum(x["first_success"] for x in rows),
             "second_attempts":sum(x["second_action_executed"] for x in rows),
             "final_successes":sum(x["final_success"] for x in rows),
             "per_task":rows,"model_optimizer_steps":0,
             "model_skill_qualified":False,
             "no_dom_oracle_for_model":True,
             "synthetic_browser_scenes":True}
    private_output_path=Path(private_output_path)
    if private_output_path.exists():
        raise ValueError("H131 second stage results must be create-only")
    private_output_path.parent.mkdir(parents=True,exist_ok=True)
    _store(private_output_path,results)
    return results

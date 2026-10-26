"""H132 owner-only paired pretrained specialist vs H131 ShowUI browser scorer.

Ground-truth / DOM target boxes are NOT used by either model or action chooser.
They are available only for optional after-inference scoring.
"""
import json
from pathlib import Path
from h130.browser_env import TASKS,BrowserDragEnv
from h131.contract import verify_manifest
from h132.grounder_contract import verify_predictions

def score_first_action(screens_dir, predictions_file, output_path):
    manifest=verify_manifest(screens_dir,expect_phase="initial")
    predictions=json.loads(Path(predictions_file).read_text("utf8"))
    samples=verify_predictions(screens_dir,predictions)
    rows=[]
    for task in TASKS:
        row=samples[task]
        with BrowserDragEnv(task) as env:
            initial=env.observe()
            original=next(c for c in manifest["cases"] if c["task"]==task)
            if initial.image_sha256 != original["image_sha256"]:
                raise ValueError("H132 actual fixture screenshot differs from model input")
            valid=row["start"] is not None and row["end"] is not None
            if valid:
                after=env.execute_drag(row["start"],row["end"])
                succeeded=after.success
                visual_change=after.observation.image_sha256!=initial.image_sha256
            else:
                succeeded=False
                visual_change=False
            rows.append({
              "task":task,"real_model_pair_valid":valid,
              "browser_drag_executed":valid,
              "actual_browser_task_success":succeeded,
              "visual_screen_changed":visual_change})
    result={
      "schema":"hala.h132.private-specialist-browser-outcomes.v1",
      "status":"H132_PRIVATE_REAL_SPECIALIST_GUI_PAIR_SCORED",
      "tasks":4,"real_model_pairs_valid":sum(x["real_model_pair_valid"] for x in rows),
      "executed_browser_drags":sum(x["browser_drag_executed"] for x in rows),
      "successes":sum(x["actual_browser_task_success"] for x in rows),
      "visual_changes":sum(x["visual_screen_changed"] for x in rows),
      "per_task":rows,
      "training_performed":False,
      "generalized_model_improvement_qualified":False,
      "source_is_same_exploratory_H131_scenes":True,
    }
    output_path=Path(output_path)
    if output_path.exists():
        raise ValueError("H132 owner evidence output create-only")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
    return result

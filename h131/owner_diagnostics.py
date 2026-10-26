"""Owner-only H131 after-inference visual grounding error audit.

Browser DOM geometry is read STRICTLY after frozen neural outputs exist and
is used only to score them. No oracle geometry enters a model prompt,
candidate generation, action execution or model evaluation input screenshot.
"""
import json
import math
import statistics
from pathlib import Path

from h130.browser_env import BrowserDragEnv, TASKS
from h131.contract import verify_manifest, verify_model_predictions

def distance_to_box(point, box):
    if (type(point) not in (list,tuple) or len(point)!=2 or
        not all(type(v) in (int,float) and math.isfinite(v) for v in point)):
        raise ValueError("H131 scoring requires a real predicted screen point")
    if any(not math.isfinite(box[name]) for name in ("x","y","width","height")):
        raise ValueError("H131 invalid actual browser locator dimensions")
    if box["width"]<=0 or box["height"]<=0:
        raise ValueError("H131 scoring requires positive true browser bbox")
    x1,y1=box["x"],box["y"]
    x2,y2=x1+box["width"],y1+box["height"]
    dx=max(x1-point[0],0,point[0]-x2)
    dy=max(y1-point[1],0,point[1]-y2)
    return math.hypot(dx,dy)

def score_points_only(source_dir, predictions_path, output_file):
    source=verify_manifest(source_dir)
    predictions=json.loads(Path(predictions_path).read_text("utf8"))
    points=verify_model_predictions(predictions,source)
    rows=[]
    for case in source["cases"]:
        task=case["task"]
        # The H130 synthetic fixture contains real browser handle bboxes.
        # Intentionally read this AFTER model predictions have been frozen.
        with BrowserDragEnv(task) as env:
            original=env.observe()
            if source["phase"]=="initial" and original.image_sha256!=case["image_sha256"]:
                raise ValueError("H131 source-to-scoring screenshot identity drift")
            handle=env._page.locator('[data-role="handle"]').bounding_box()
            target=env._page.locator(".target").bounding_box()
            if handle is None or target is None:
                raise ValueError("H131 actual Chromium fixture control missing")
            start_distance=distance_to_box(points[task]["start"],handle)
            end_distance=distance_to_box(points[task]["end"],target)
            rows.append({
                "task":task,
                "start_hits_actual_draggable":start_distance==0,
                "start_pixel_distance_to_handle":round(start_distance,2),
                "end_pixel_distance_to_green_region":round(end_distance,2),
            })
    result={
        "schema":"hala.h131.owner-only-browser-grounding-error.v1",
        "source_phase":source["phase"],
        "real_pretrained_neural_cases":4,
        "actual_draggable_handle_hits":sum(r["start_hits_actual_draggable"] for r in rows),
        "median_start_pixel_error_to_handle":statistics.median(
            r["start_pixel_distance_to_handle"] for r in rows),
        "median_end_pixel_error_to_target_region":statistics.median(
            r["end_pixel_distance_to_green_region"] for r in rows),
        "per_task":rows,
        "model_skill_qualified":False,
        "oracle_geometry_exposed_to_model":False,
        "model_training_performed":False,
    }
    output_file=Path(output_file)
    if output_file.exists():
        raise ValueError("H131 owner-private after-inference scoring output create-only")
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result

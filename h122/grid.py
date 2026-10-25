"""H122 absolute END region+precise-point objective, strictly original TRAIN.

No oracle END coordinates are ever inserted into an inference prompt.
Every output includes a 10x10 cell consistent with true H113 normalized END.
"""
from __future__ import annotations
import json
from h61.official_eval import strict_json
from h113.dual_point import relative_to_source_pixel, valid_box

SCHEMA="hala.h122.original-end-absolute-grid10-finepoint.v1"
SPACE="normalized_0_1000"
GRID_SIZE=10

def grid_cell(point):
    if (type(point) is not list or len(point)!=2 or any(
            type(v) is not int or not 0<=v<=1000 for v in point)):
        raise ValueError("original H113 normalized END point required")
    return [min(9,point[0]//100),min(9,point[1]//100)]

def end_prompt(intent,ordered):
    if (type(intent) is not str or not 4<=len(intent)<=1200
            or type(ordered) is not bool):
        raise ValueError("original bounded source task and direction required")
    return (
        f"{intent}\n"
        "Locate ONLY the DESTINATION (END) of the requested drag in the "
        "ORIGINAL full screenshot. Ignore the initial drag handle. "
        "Use normalized image coordinates 0 to 1000 left-to-right and "
        "top-to-bottom, not screenshot pixel values. Divide the image "
        "conceptually into a 10 by 10 grid: column=floor(x/100), "
        "row=floor(y/100), capped at 9. "
        "First identify coarse cell [column,row], then precise END point "
        "[integer_x,integer_y]. The cell MUST contain the point. "
        'Output ONLY strict JSON with exactly keys "role","cell","point",'
        '"coordinate_space": role="end", coordinate_space="normalized_0_1000". '
        "Do not execute any action."
    )

def end_completion(point):
    return json.dumps({"role":"end","cell":grid_cell(point),
                       "point":point,"coordinate_space":SPACE},
                      sort_keys=True,separators=(",",":"))

def parse_end(raw,w,h):
    """Fail closed on off-image, malformed, repeated or cell-inconsistent JSON."""
    if type(raw) is not str or not 1<=len(raw)<=8192:
        return None
    try:
        x=strict_json(raw)
    except (TypeError,ValueError):
        return None
    if (type(x) is not dict or set(x)!={"role","cell","point","coordinate_space"}
            or x["role"]!="end" or x["coordinate_space"]!=SPACE):
        return None
    try:
        cell=grid_cell(x["point"])
        if (type(x["cell"]) is not list or len(x["cell"])!=2
                or any(type(v) is not int or not 0<=v<=9 for v in x["cell"])
                or cell!=x["cell"]):
            return None
        return relative_to_source_pixel(x["point"],w,h)
    except (TypeError,ValueError):
        return None

def overlay(original,point_row):
    """Source-admit an unchanged original H113 END role; data only."""
    from h113.dual_point import COORDINATE_SPACE
    if (type(original) is not dict or type(point_row) is not dict
            or original.get("split")!="train"
            or point_row.get("split")!="train"
            or point_row.get("original_source_split")!="train"
            or point_row.get("endpoint_role")!="end"
            or point_row.get("no_eval_or_test_source") is not True
            or point_row.get("source_bound_validation_passed") is not True):
        raise ValueError("H122 requires original H113 TRAIN-only authentic END")
    for key in ("id","image_sha256","image_path","language","family"):
        original_key="original_source_id" if key=="id" else key
        if original.get(key)!=point_row.get(original_key):
            raise ValueError("H122 exact original H71 and H113 source identity differs")
    w,h=original["image_width"],original["image_height"]
    if (w!=point_row["source_image_width"] or h!=point_row["source_image_height"]):
        raise ValueError("H122 original screenshot spatial dimensions differ")
    try:
        end=strict_json(point_row["completion"])
    except (TypeError,ValueError) as e:
        raise ValueError("H122 untrusted original END completion") from e
    if (type(end) is not dict or set(end)!={"role","point","coordinate_space"}
            or end["role"]!="end" or end["coordinate_space"]!=COORDINATE_SPACE):
        raise ValueError("H122 original END role target drift")
    px=relative_to_source_pixel(end["point"],w,h)
    if (not valid_box(original["end_bbox"],w,h)
            or not (original["end_bbox"][0]<=px[0]<=original["end_bbox"][2]
                    and original["end_bbox"][1]<=px[1]<=original["end_bbox"][3])):
        raise ValueError("H122 original TRAIN END target escapes true original box")
    # Derive original intent from H113 endpoint role prompt; no EVAL/task boxes.
    prompt=point_row["prompt"]
    marker="\nLocate ONLY the END endpoint"
    if type(prompt) is not str or prompt.count(marker)!=1:
        raise ValueError("H122 must use original H113 immutable source intent")
    intent=prompt.split(marker)[0]
    result=dict(original)
    result.update(
        id=original["id"]+"::h122_end",
        endpoint_role="end",
        prompt=end_prompt(intent,original["ordered"]),
        completion=end_completion(end["point"]))
    return result


def verify_overlay(original,derived):
    """Assert source H71 original TRAIN image and END target consistency."""
    if original.get("split")!="train" or derived.get("split")!="train":
        raise ValueError("H122 TRAIN scope required")
    if derived.get("original_source_id")!=original.get("id"):
        raise ValueError("H122 source ID drift")
    for key in ("image_path","image_sha256","family","language"):
        if original.get(key)!=derived.get(key):
            raise ValueError("H122 source media mismatch")
    if derived.get("endpoint_role")!="end" or derived.get("schema")!=SCHEMA:
        raise ValueError("H122 only isolated absolute END")
    w,h=original["image_width"],original["image_height"]
    if (w!=derived.get("source_image_width") or h!=derived.get("source_image_height")
            or not valid_box(original["end_bbox"],w,h)):
        raise ValueError("H122 genuine source END box required")
    point=parse_end(derived["completion"],w,h)
    box=original["end_bbox"]
    if point is None or not (box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]):
        raise ValueError("H122 genuine source-bound END target required")
    return True

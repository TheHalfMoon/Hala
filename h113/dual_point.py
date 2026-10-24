"""H113 original DragOn TRAIN only, source-authorized separate point supervision.

NO new model training, no original media mutation and no EVAL/TEST examples.
Turns one 2-endpoint original source drag into two distinct *independent*
original-image point localization SFT completions with Qwen relative units.
"""
from __future__ import annotations
import json
from h61.prompt_contract import source_body,drag_prompt
from h61.official_eval import strict_json

SCHEMA="hala.h113.original-train-frozen-two-points-relative1000.v1"
COORDINATE_SPACE="normalized_0_1000"
ROLES=("start","end")


class QuantizationQuarantine(ValueError):
    """Source is genuine TRAIN, but normalizing moves a point outside its box."""


def valid_box(box,w,h):
    return (type(box) is list and len(box)==4
            and all(type(x) is int for x in box)
            and 0<=box[0]<=box[2]<w and 0<=box[1]<=box[3]<h)


def source_pixel_to_relative(point,w,h):
    if (type(point) is not list or len(point)!=2
            or any(type(v) is not int for v in point)
            or not 0<=point[0]<w or not 0<=point[1]<h):
        raise ValueError("original source image pixel point out of bounds")
    return [round(point[0]*1000/w),round(point[1]*1000/h)]


def relative_to_source_pixel(point,w,h):
    if (type(point) is not list or len(point)!=2
            or any(type(v) is not int or not 0<=v<=1000 for v in point)):
        raise ValueError("strict Qwen relative pair needed")
    return [min(w-1,round(point[0]*w/1000)),
            min(h-1,round(point[1]*h/1000))]


def point_prompt(intent:str,ordered:bool,role:str)->str:
    if (type(intent) is not str or not 4<=len(intent)<=1200
            or type(ordered) is not bool or role not in ROLES):
        raise ValueError("source original intent, direction and endpoint required")
    return (
        f"{intent}\n"
        f"Locate ONLY the {role.upper()} endpoint of the requested drag in "
        "the ORIGINAL screenshot. Do not predict both endpoints. "
        "Use Qwen relative [0,1000] image coordinates: x from left to right, "
        "y from top to bottom. The image is not cropped, resized or warped. "
        "Output ONLY exact JSON with keys point, role, coordinate_space. "
        f'role="{role}", coordinate_space="{COORDINATE_SPACE}", '
        "point=[integer_x,integer_y]. Do not execute any actions."
    )


def admit_original_source(row:dict,original:dict):
    if (type(row) is not dict or type(original) is not dict
            or row.get("split")!="train" or original.get("split")!="train"
            or original.get("action_type")!="drag"
            or row.get("id")!=original.get("id")
            or row.get("image_sha256")!=original.get("image_sha256")
            or row.get("image_path")!=original.get("image_path")
            or row.get("language") not in ("ar","en")
            or not row.get("family") or not row.get("id")
            or type(original.get("ordered")) is not bool):
        raise ValueError("H113 requires exact original H71 source TRAIN identity")
    w,h=original["image_width"],original["image_height"]
    if (type(w) is not int or type(h) is not int
            or not 64<=w<=1024 or not 64<=h<=640):
        raise ValueError("original source H61 frame bounds violated")
    original_prompt=drag_prompt(source_body(original["intent"],original["ordered"]),
                                w,h,original["ordered"])
    if row.get("prompt")!=original_prompt:
        raise ValueError("H113 original H61 prompt identity drift")
    action=strict_json(row["completion"])
    if (action!=original["completion"]
            or set(action)!={"type","start_point","end_point","coordinate_space","ordered"}
            or action["type"]!="drag"
            or action["coordinate_space"]!="source_image_pixels"
            or action["ordered"]!=original["ordered"]):
        raise ValueError("original full action target was altered")
    for role in ROLES:
        box=original[role+"_bbox"]
        if not valid_box(box,w,h):
            raise ValueError("invalid ORIGINAL TRAIN target rectangle")
        pixel=action[role+"_point"]
        if (type(pixel) is not list or len(pixel)!=2
                or any(type(v) is not int for v in pixel)
                or pixel!=[(box[0]+box[2])//2,(box[1]+box[3])//2]):
            raise ValueError("H113 target not original verified box center")
    return w,h,action


def derive_two_independent_points(row:dict,original:dict)->tuple[dict,dict]:
    """Reject source-border/out-of-box normalization, never shift oracle GT."""
    w,h,action=admit_original_source(row,original)
    points=[]
    for role in ROLES:
        original_pixel=action[role+"_point"]
        relative=source_pixel_to_relative(original_pixel,w,h)
        pixel_round_trip=relative_to_source_pixel(relative,w,h)
        box=original[role+"_bbox"]
        if not (box[0]<=pixel_round_trip[0]<=box[2]
                and box[1]<=pixel_round_trip[1]<=box[3]):
            raise QuantizationQuarantine("Qwen normalized quantization escapes source box; quarantine source")
        completion=json.dumps(
            {"point":relative,"role":role,"coordinate_space":COORDINATE_SPACE},
            sort_keys=True,separators=(",",":"))
        points.append({
            "schema":SCHEMA,"id":row["id"]+"::"+role,"original_source_id":row["id"],
            "split":"train","original_source_split":"train",
            "language":row["language"],"family":row["family"],
            "endpoint_role":role,"source_image_width":w,"source_image_height":h,
            "image_path":row["image_path"],"image_sha256":row["image_sha256"],
            "prompt":point_prompt(original["intent"],original["ordered"],role),
            "completion":completion,"source_bound_validation_passed":True,
            "no_eval_or_test_source":True
        })
    if points[0]["id"]==points[1]["id"] or points[0]["completion"]==points[1]["completion"]:
        raise ValueError("source does not provide two distinct independent point targets")
    return points[0],points[1]

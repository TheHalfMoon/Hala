"""H124 both original START + END, one model prediction, no oracle anchor.

Original H71 TRAIN-only targets. Avoid H117/H119 mismatch between
oracle START at training and model-produced START at inference:
start/end/delta are predicted together from the SAME genuine screenshot.
"""
from __future__ import annotations
import json
from h61.official_eval import strict_json
from h113.dual_point import (derive_two_independent_points,
                             relative_to_source_pixel,QuantizationQuarantine)

SCHEMA="hala.h124.original-complex-TRAIN-joint-start-end-delta.v1"
SPACE="normalized_0_1000"


def cell(point):
    if (type(point) is not list or len(point)!=2 or
            any(type(x) is not int or not 0<=x<=1000 for x in point)):
        raise ValueError("H124 exact normalized START or END point required")
    return [min(9,point[0]//100),min(9,point[1]//100)]


def completion(start,end):
    a=cell(start);b=cell(end)
    delta=[end[0]-start[0],end[1]-start[1]]
    return json.dumps({
       "start_point":start,"end_point":end,
       "start_cell":a,"end_cell":b,"delta":delta,
       "coordinate_space":SPACE},sort_keys=True,separators=(",",":"))


def prompt(intent,ordered):
    if (type(intent) is not str or not 4<=len(intent)<=1200
            or type(ordered) is not bool):
        raise ValueError("H124 original TRAIN source intent needed")
    return (f"{intent}\nFrom the ORIGINAL unmodified screenshot, "
            "locate the source/START UI element for this drag and the "
            "destination/END UI element. Predict BOTH points yourself, "
            "not an oracle anchor. Report exact normalized 0..1000 "
            "image START and END coordinates, their two 10x10 grid cells, "
            "and signed delta = END minus START. The cells must contain "
            "the respective points and START+delta must equal END. "
            'Output only exact JSON keys "start_point","end_point",'
            '"start_cell","end_cell","delta","coordinate_space". '
            "Do not execute or click.")


def parse(raw,w,h):
    if (type(raw) is not str or not 1<=len(raw)<=8192
            or type(w) is not int or type(h) is not int
            or not 64<=w<=1024 or not 64<=h<=640):
        return None
    try:
        obj=strict_json(raw)
        if (type(obj) is not dict or set(obj)!={
            "start_point","end_point","start_cell","end_cell",
            "delta","coordinate_space"} or
                obj["coordinate_space"]!=SPACE or
                type(obj["start_cell"]) is not list or
                type(obj["end_cell"]) is not list or
                type(obj["delta"]) is not list or
                len(obj["delta"])!=2 or
                any(type(x) is not int or not -1000<=x<=1000
                    for x in obj["delta"])):
            return None
        start,end=obj["start_point"],obj["end_point"]
        if (obj["start_cell"]!=cell(start) or
                obj["end_cell"]!=cell(end) or
                obj["delta"]!=[end[0]-start[0],end[1]-start[1]]):
            return None
        return (relative_to_source_pixel(start,w,h),
                relative_to_source_pixel(end,w,h))
    except (ValueError,TypeError,KeyError):
        return None


def from_original(sft,source):
    start,end=derive_two_independent_points(sft,source)
    a=strict_json(start["completion"])["point"]
    b=strict_json(end["completion"])["point"]
    role={
      "schema":SCHEMA,"id":sft["id"]+"::h124_both",
      "original_source_id":sft["id"],"split":"train",
      "original_source_split":"train",
      "language":sft["language"],"family":sft["family"],
      "image_path":sft["image_path"],"image_sha256":sft["image_sha256"],
      "source_image_width":source["image_width"],
      "source_image_height":source["image_height"],
      "endpoint_role":"joint_start_end",
      "prompt":prompt(source["intent"],source["ordered"]),
      "completion":completion(a,b),"source_bound_validation_passed":True,
      "no_eval_or_test_source":True}
    pair=parse(role["completion"],source["image_width"],source["image_height"])
    if pair is None:
        raise QuantizationQuarantine("H124 original paired points cannot be reconstructed")
    for i,name in enumerate(("start_bbox","end_bbox")):
        x,y=pair[i]; box=source[name]
        if not (box[0]<=x<=box[2] and box[1]<=y<=box[3]):
            raise QuantizationQuarantine("H124 original START/END quantization exceeds genuine source bbox")
    # Prompt uses source intent and ordered instruction only; original
    # image target boxes and START/END labels remain in completion alone.
    return role

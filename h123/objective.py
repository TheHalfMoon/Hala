"""H123 complex GUI END: original TRAIN-only coarse 4x4, fine 10x10, point.

No source bbox or oracle endpoint enters a model prompt. Target points
are derived solely from original H71 source TRAIN actions with H113's
unchanged quantization and exact source bounding-box admission.
"""
from __future__ import annotations
import json
from h113.dual_point import (derive_two_independent_points,
                             relative_to_source_pixel)
from h61.official_eval import strict_json
from h61.prompt_contract import source_body

SCHEMA="hala.h123.complex-original-TRAIN-END-dual-grid-point.v1"
SPACE="normalized_0_1000"


def grids(point):
    if (type(point) is not list or len(point)!=2 or
            any(type(x) is not int or not 0<=x<=1000 for x in point)):
        raise ValueError("H123 valid original normalized point is required")
    return ([min(3,point[0]//250),min(3,point[1]//250)],
            [min(9,point[0]//100),min(9,point[1]//100)])


def completion(point):
    quadrant,cell=grids(point)
    return json.dumps({
        "role":"end","quadrant":quadrant,"cell":cell,"point":point,
        "coordinate_space":SPACE},sort_keys=True,separators=(",",":"))


def prompt(intent,ordered):
    if (type(intent) is not str or not 4<=len(intent)<=1200
            or type(ordered) is not bool):
        raise ValueError("H123 genuine original TRAIN intent required")
    return (
        f"{intent}\nLocate ONLY the destination (END) target for the "
        "requested drag in the original uncropped GUI screenshot. "
        "First identify a coarse 4x4 grid quadrant [col,row], then "
        "a 10x10 grid cell [col,row] and the precise normalized END "
        "point [x,y] in 0..1000 original-image coordinates. "
        "Column increases right; row increases down. The quadrant "
        "and cell MUST both contain the same precise point. "
        'Output ONLY exact JSON with keys "role","quadrant",'
        '"cell","point","coordinate_space": role="end", '
        'coordinate_space="normalized_0_1000". Do not act.'
    )


def parse(raw,width,height):
    if (type(raw) is not str or not 1<=len(raw)<=8192
            or type(width) is not int or type(height) is not int
            or not 64<=width<=1024 or not 64<=height<=640):
        return None
    try:
        obj=strict_json(raw)
        if (type(obj) is not dict or set(obj)!={"role","quadrant","cell",
                "point","coordinate_space"} or obj["role"]!="end"
                or obj["coordinate_space"]!=SPACE or
                type(obj["quadrant"]) is not list or
                type(obj["cell"]) is not list):
            return None
        quadrant,cell=grids(obj["point"])
        if obj["quadrant"]!=quadrant or obj["cell"]!=cell:
            return None
        return relative_to_source_pixel(obj["point"],width,height)
    except (ValueError,TypeError,KeyError):
        return None


def from_original(sft,original):
    """Original H113 TRAIN dual-point admission; return END-only H123."""
    start,end=derive_two_independent_points(sft,original)
    answer=strict_json(end["completion"])
    point=answer["point"]
    result=dict(end)
    result.update(schema=SCHEMA,id=sft["id"]+"::h123_end",
                  endpoint_role="end",
                  prompt=prompt(original["intent"],original["ordered"]),
                  completion=completion(point))
    if (result["image_sha256"]!=sft["image_sha256"] or
            result["original_source_id"]!=sft["id"] or
            parse(result["completion"],original["image_width"],
                  original["image_height"]) is None):
        raise ValueError("H123 original END geometry/source identity mismatch")
    return result

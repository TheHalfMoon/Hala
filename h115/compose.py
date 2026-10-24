"""H115 strict H113 dual-point source-pixel endpoint adapter for new GUI EVAL.

Never repairs invalid JSON, invents hidden target positions, or relaxes
the original H61 native drag task scorer. Separate original image pass
per START and END costs twice the model calls.
"""
import json
from h61.official_eval import strict_json,parse_action
from h113.dual_point import (COORDINATE_SPACE,ROLES,
                              relative_to_source_pixel,point_prompt)

SCHEMA="hala.h115.two-independent-native-normalized-points-original-H61-drag.v1"

def role_input(original:dict,role:str)->dict:
    """Use original task intent/direction, never the reference target boxes."""
    if role not in ROLES or type(original) is not dict:
        raise ValueError("exact START or END original GUI source role required")
    task=original["task"]
    clone=dict(original)
    clone["prompt"]=point_prompt(task["intent"],task["ordered"],role)
    return clone


def parse_point(raw:str,role:str,width:int,height:int)->list[int]|None:
    """Strict Qwen relative native original-image POINT or None."""
    if not isinstance(raw,str) or not 1<=len(raw)<=8192 or role not in ROLES:
        return None
    try:
        item=strict_json(raw)
    except (ValueError,TypeError):
        return None
    if (type(item) is not dict or set(item)!={"point","role","coordinate_space"}
            or item.get("role")!=role
            or item.get("coordinate_space")!=COORDINATE_SPACE):
        return None
    try:
        return relative_to_source_pixel(item["point"],width,height)
    except (TypeError,ValueError):
        return None


def compose_strict_original_drag(original:dict,start_raw:str,end_raw:str)->str|None:
    """Round-trip both independently generated points into immutable H61."""
    width,height=original["image_width"],original["image_height"]
    a=parse_point(start_raw,"start",width,height)
    b=parse_point(end_raw,"end",width,height)
    if a is None or b is None:return None
    action={"type":"drag","start_point":a,"end_point":b,
            "ordered":original["task"]["ordered"],
            "coordinate_space":"source_image_pixels"}
    encoded=json.dumps(action,sort_keys=True,separators=(",",":"))
    if parse_action(encoded,width,height)!=action:
        return None
    return encoded

"""H112 Qwen3-VL native [0,1000] relative → strict original H61 pixel contract.

Upstream Qwen3-VL 2D grounding uses 0..1000 relative positions; H61 scorer
expects source-image pixels. NO source GT boxes, hidden TEST or weak JSON
repair are permitted in this deterministic conversion.
"""
from __future__ import annotations
import json
from h61.official_eval import STRICT_FIELDS,strict_json,parse_action

NATIVE_SPACE="normalized_0_1000"
SCHEMA="hala.h112.qwen-relative-0-to-1000-source-pixel-bridge.v1"

def normalized_prompt(scene:dict)->str:
    """Original source instruction, output contract only; never GT coordinates."""
    width,height=scene["image_width"],scene["image_height"]
    task=scene["task"]
    intent=task["intent"]
    ordered=task["ordered"]
    if (type(width) is not int or type(height) is not int
            or not 64<=width<=1024 or not 64<=height<=640
            or not isinstance(intent,str) or not 4<=len(intent)<=1200
            or type(ordered) is not bool):
        raise ValueError("source-bound native relative Qwen prompt required")
    o="true" if ordered else "false"
    return (f"{intent}\n"
       "You are pointing to exact elements on the provided GUI screenshot. "
       "Find the start of the requested drag and its destination. "
       "Use Qwen3-VL's native NORMALIZED relative coordinate plane [0,1000]: "
       "x=0 means left, x=1000 means right, y=0 means top, y=1000 means bottom. "
       "Do NOT output image pixels, do NOT describe your reasoning. "
       "Return ONLY a JSON object with EXACTLY five keys: "
       "type=\"drag\", start_point=[x,y], end_point=[x,y], "
       f"coordinate_space=\"{NATIVE_SPACE}\", ordered={o}. "
       "Both start and end points must be integer pairs in 0..1000. "
       "Do not execute any mouse actions.")

def convert(raw:str,width:int,height:int)->str|None:
    """Return exact original H61 native JSON iff strict relative JSON valid."""
    if (not isinstance(raw,str) or not 1<=len(raw)<=8192
            or type(width) is not int or type(height) is not int
            or width<=0 or height<=0 or width>1024 or height>640):
        return None
    try:
        o=strict_json(raw)
    except (ValueError,TypeError):
        return None
    if (type(o) is not dict or set(o)!=STRICT_FIELDS
            or o.get("type")!="drag" or o.get("coordinate_space")!=NATIVE_SPACE
            or type(o.get("ordered")) is not bool):
        return None
    result={"type":"drag","coordinate_space":"source_image_pixels",
            "ordered":o["ordered"]}
    for k in ("start_point","end_point"):
        xy=o.get(k)
        if (type(xy) is not list or len(xy)!=2 or
                any(type(v) is not int or not 0<=v<=1000 for v in xy)):
            return None
        # Official Qwen normalized convention uses /1000 * source width.
        # Exact right/bottom boundaries are clipped inside valid source.
        result[k]=[
            min(width-1,round(xy[0]*width/1000)),
            min(height-1,round(xy[1]*height/1000))]
    output=json.dumps(result,sort_keys=True,separators=(",",":"))
    if parse_action(output,width,height)!=result:return None
    return output

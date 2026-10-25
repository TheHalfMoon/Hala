"""H125 separate pretrained GUI-grounding 2B native output contracts.

UGround: parenthesized coordinates normalized 0..1000.
ShowUI: [x,y] or (x,y) normalized 0..1.
NEVER silently reinterpret 0..1 and 0..1000.
"""
from __future__ import annotations
import ast
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Model:
    model_id:str
    revision_sha:str
    license:str
    coordinate_space:str
    arch:str
    pretrained_gui_grounding:bool

MODELS={
  "uground_2b":Model("osunlp/UGround-V1-2B",
                     "ea7109f9573e5d8ee5170e425bd1fd73f2020e06",
                     "apache-2.0","normalized_0_1000","qwen2_vl",True),
  "showui_2b":Model("showlab/ShowUI-2B",
                    "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60",
                    "mit","relative_0_1","qwen2_vl",True)
}
UGROUND_PROMPT="""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of interest.
Description: {description}
Answer:"""
SHOWUI_SYSTEM="Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

def prompt_for(model:str,intent:str,role:str):
    if model not in MODELS or role not in ("START","END"):
        raise ValueError("H125 pinned owner two-model/role experiment only")
    if not isinstance(intent,str) or not 4<=len(intent)<=1200:
        raise ValueError("H125 genuine original source TRAIN instruction required")
    description=intent+" Locate the "+role+" handle/target of this drag."
    return (UGROUND_PROMPT.format(description=description)
            if model=="uground_2b"
            else SHOWUI_SYSTEM+"\n"+description)

def parse_point(model:str,raw:str,w:int,h:int):
    if (model not in MODELS or type(raw) is not str or not 1<=len(raw)<=128
            or type(w) is not int or type(h) is not int
            or not 64<=w<=1024 or not 64<=h<=640):
        return None
    try:
        xy=ast.literal_eval(raw.strip())
    except (ValueError,SyntaxError,TypeError,MemoryError,RecursionError):
        return None
    if (type(xy) not in (tuple,list) or len(xy)!=2 or
            any(type(v) not in (int,float) or
                not math.isfinite(v) for v in xy)):
        return None
    if model=="uground_2b" and (type(xy) is not tuple or
             any(type(val) is not int for val in xy)):
        return None
    if model=="showui_2b" and type(xy) is not list:
        return None
    maximum=1000 if model=="uground_2b" else 1
    if any(not 0<=val<=maximum for val in xy):
        return None
    return (min(w-1,round(xy[0]*w/maximum)),
            min(h-1,round(xy[1]*h/maximum)))

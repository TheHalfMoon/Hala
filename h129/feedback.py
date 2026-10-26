"""H129 one-step visual guessed-cursor marker feedback, not actual desktop interaction.

The marker is drawn strictly from a first-pass frozen-model prediction.
No oracle labels, target boxes, previous heldouts or automated actions.
"""
import math
from collections import Counter
from PIL import ImageDraw
from h125.model_contracts import SHOWUI_SYSTEM

def validate_point(point,w,h):
    if (type(w) is not int or type(h) is not int or w<64 or h<64 or
        type(point) not in (tuple,list) or len(point)!=2 or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in point) or
        not (0<=point[0]<w and 0<=point[1]<h)):
        raise ValueError("H129 genuine in-bounds model point needed")
    return tuple(point)

def draw_marker(image,point):
    """Return modified copy, preserving native dimensions and original input."""
    if image.mode!="RGB":
        raise ValueError("H129 requires actual original RGB image")
    x,y=validate_point(point,image.width,image.height)
    result=image.copy()
    painter=ImageDraw.Draw(result)
    x=int(round(x));y=int(round(y))
    radius=7
    painter.ellipse((x-radius,y-radius,x+radius,y+radius),
                    outline=(255,0,210),width=2)
    painter.line((x-10,y,x-4,y),fill=(255,0,210),width=2)
    painter.line((x+4,y,x+10,y),fill=(255,0,210),width=2)
    painter.line((x,y-10,x,y-4),fill=(255,0,210),width=2)
    painter.line((x,y+4,x,y+10),fill=(255,0,210),width=2)
    return result

def feedback_prompt(intent,role):
    if (role not in ("start","end") or not isinstance(intent,str)
        or not 4<=len(intent)<=1200):
        raise ValueError("H129 genuine START/END instruction and role needed")
    task=(f"{intent} Locate the {role.upper()} handle/target of this drag. "
          "The magenta ring and crosshair is a previous model GUESS at this "
          "location, not evidence of the target or the ground truth. "
          "Reinspect the screenshot and decide whether to correct the guess. "
          "Respond with one precise clickable location [x, y] on the entire "
          "image in relative [0,1] coordinates, without explanation.")
    return SHOWUI_SYSTEM+"\n"+task

def hit(point,bbox):
    return bbox[0]<=point[0]<=bbox[2] and bbox[1]<=point[1]<=bbox[3]

def score(cases,predictions):
    if len(cases)!=16 or len(predictions)!=16:
        raise ValueError("H129 exact paired original TRAIN16 required")
    totals=Counter()
    families={}
    seen=set()
    for case,pred in zip(cases,predictions):
        if (case["id"]!=pred.get("id") or case["image_sha256"]!=pred.get("image_sha256")
            or case["id"] in seen):
            raise ValueError("H129 source / raw inference identity mismatch")
        seen.add(case["id"])
        f=case["family"]
        family=families.setdefault(f,{"n":0,"direct_full":0,"feedback_full":0})
        family["n"]+=1
        flags={}
        for role in ("start","end"):
            d=validate_point(pred["direct_"+role],case["image_width"],case["image_height"])
            p=validate_point(pred["feedback_"+role],case["image_width"],case["image_height"])
            bbox=case[role+"_bbox"]
            dc=hit(d,bbox); fc=hit(p,bbox)
            totals["direct_"+role]+=int(dc)
            totals["feedback_"+role]+=int(fc)
            totals["changed_"+role]+=int(d!=p)
            cx=(bbox[0]+bbox[2])/2;cy=(bbox[1]+bbox[3])/2
            totals["direct_quadrant_"+role]+=int(
                (d[0]>=case["image_width"]/2)==(cx>=case["image_width"]/2) and
                (d[1]>=case["image_height"]/2)==(cy>=case["image_height"]/2))
            totals["feedback_quadrant_"+role]+=int(
                (p[0]>=case["image_width"]/2)==(cx>=case["image_width"]/2) and
                (p[1]>=case["image_height"]/2)==(cy>=case["image_height"]/2))
            flags[role]=(dc,fc)
        d_full=flags["start"][0] and flags["end"][0]
        f_full=flags["start"][1] and flags["end"][1]
        totals["direct_full"]+=int(d_full);totals["feedback_full"]+=int(f_full)
        family["direct_full"]+=int(d_full);family["feedback_full"]+=int(f_full)
    if len(families)!=4 or any(x["n"]!=4 for x in families.values()):
        raise ValueError("H129 exactly four original TRAIN scenes per family required")
    return {"schema":"hala.h129.model-guessed-cursor-marker-revision16.v1",
            "status":"EXPLORATORY_PAIRED_ONLY_NOT_MODEL_QUALIFICATION",
            "cases":16,"aggregate":dict(totals),"per_family":families,
            "model_skill_qualified":False,"new_optimizer_steps":0,
            "original_EVAL_TEST_used":False,
            "is_real_interactive_environment":False,
            "no_oracle_ground_truth_in_feedback":True}

"""H128 leak-free two-pass zoom/refine inference geometry and diagnostics.

Crop center is derived *only* from the pretrained baseline output, never
the actual target bbox. Refine the SAME frozen model on crop; reproject native
crop predictions back to original screenshot space.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Crop:
    left:int
    top:int
    right:int
    bottom:int

    @property
    def width(self):
        return self.right-self.left

    @property
    def height(self):
        return self.bottom-self.top

def anchored_crop(width:int,height:int,point,ratio:float=0.5):
    if (type(width) is not int or type(height) is not int or
        not 64<=width<=1024 or not 64<=height<=640 or
        type(ratio) is not float or not 0.2<=ratio<=0.9 or
        type(point) not in (tuple,list) or len(point)!=2 or
        any(type(p) not in (int,float) or not math.isfinite(p) for p in point) or
        not 0<=point[0]<width or not 0<=point[1]<height):
        raise ValueError("H128 crop requires in-bounds pretrained native pixel point")
    cw=max(64,round(width*ratio))
    ch=max(64,round(height*ratio))
    cw=min(width,cw);ch=min(height,ch)
    left=max(0,min(width-cw,round(point[0]-cw/2)))
    top=max(0,min(height-ch,round(point[1]-ch/2)))
    return Crop(left,top,left+cw,top+ch)

def native_to_original(crop:Crop,point,source_width:int,source_height:int):
    if (not 0<=crop.left<crop.right<=source_width or
        not 0<=crop.top<crop.bottom<=source_height or
        type(point) not in (tuple,list) or len(point)!=2 or
        any(type(p) not in (int,float) or not math.isfinite(p) for p in point) or
        not 0<=point[0]<crop.width or not 0<=point[1]<crop.height):
        raise ValueError("H128 native crop-to-original coordinate out of range")
    return (crop.left+round(point[0]),crop.top+round(point[1]))

def box_hit(point,bbox):
    return (point is not None and bbox[0]<=point[0]<=bbox[2] and
            bbox[1]<=point[1]<=bbox[3])

def score(cases, predictions):
    if len(cases)!=16 or len(predictions)!=16:
        raise ValueError("H128 exact frozen 16 paired original-source scenes required")
    totals={"direct_start":0,"direct_end":0,"direct_full":0,
            "zoom_start":0,"zoom_end":0,"zoom_full":0,
            "coarse_in_quadrant_start":0,"coarse_in_quadrant_end":0,
            "refine_calls":0,"crop_includes_target_start":0,
            "crop_includes_target_end":0}
    per_family={}
    seen=set()
    for case,pred in zip(cases,predictions):
        if (case["id"]!=pred.get("id") or case["image_sha256"]!=pred.get("image_sha256")
                or case["id"] in seen):
            raise ValueError("H128 result/source identity mismatch")
        seen.add(case["id"])
        family=case["family"]
        metrics=per_family.setdefault(family,{"n":0,"direct_full":0,"zoom_full":0})
        metrics["n"]+=1
        hit={}
        for role in ("start","end"):
            bbox=case[role+"_bbox"]
            direct=pred.get("direct_"+role)
            zoom=pred.get("zoom_"+role)
            crop=pred.get("crop_"+role)
            if direct is None or type(direct) not in (tuple,list) or len(direct)!=2:
                raise ValueError("H128 pretrained direct coordinate missing")
            if not all(type(x) in (int,float) and math.isfinite(x) for x in direct):
                raise ValueError("H128 invalid direct coordinates")
            w=case["image_width"];h=case["image_height"]
            if not (0<=direct[0]<w and 0<=direct[1]<h):
                raise ValueError("H128 direct point out of screenshot")
            dx=box_hit(direct,bbox)
            zx=box_hit(zoom,bbox)
            totals["direct_"+role]+=int(dx)
            totals["zoom_"+role]+=int(zx)
            tgt=((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
            totals["coarse_in_quadrant_"+role]+=int(
                int(direct[0]>=w/2)==int(tgt[0]>=w/2) and
                int(direct[1]>=h/2)==int(tgt[1]>=h/2))
            if crop is not None:
                if type(crop) is not Crop:
                    raise ValueError("H128 crop must be original-source anchored")
                totals["refine_calls"]+=1
                totals["crop_includes_target_"+role]+=int(
                    crop.left<=tgt[0]<crop.right and crop.top<=tgt[1]<crop.bottom)
            hit[role]=(dx,zx)
        direct_full=hit["start"][0] and hit["end"][0]
        zoom_full=hit["start"][1] and hit["end"][1]
        totals["direct_full"]+=int(direct_full)
        totals["zoom_full"]+=int(zoom_full)
        metrics["direct_full"]+=int(direct_full)
        metrics["zoom_full"]+=int(zoom_full)
    if len(per_family)!=4 or any(v["n"]!=4 for v in per_family.values()):
        raise ValueError("H128 exact four source families x four original scenes")
    return {
        "schema":"hala.h128.private-exploratory-pretrained-zoom-16.v1",
        "status":"EXPLORATORY_PAIRED_ONLY_NOT_MODEL_QUALIFICATION",
        "cases":16,"per_family":per_family,"aggregate":totals,
        "model_accuracy_proven":False,
        "new_optimizer_steps":0,"source_only_original_TRAIN":True,
        "do_not_compare_to_H125_scene_set":True,
    }

"""H110 independent coarse 1-pass + bounded local crop refinement controls.

Coordinates remain in original H61 source image frame for final scoring.
The model's own first-pass point (not target bbox/oracle) selects each crop.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from h61.official_eval import build_prompt,parse_action
from h109.crop_geometry import crop_around_point
from training.artifacts import sha256_file

ZOOM_SIZE=512


@dataclass(frozen=True)
class Focus:
    scene:dict
    frame:object
    endpoint:str


def make_focus(original:dict,coarse:dict,endpoint:str,target:Path):
    if endpoint not in ("start","end"):
        raise ValueError("one of the original drag endpoints required")
    width,height=original["image_width"],original["image_height"]
    if not isinstance(coarse,dict) or coarse.get("type")!="drag":
        raise ValueError("only a native valid first-pass model drag accepted")
    point=coarse["start_point" if endpoint=="start" else "end_point"]
    with Image.open(original["image_path"]) as image:
        if image.size!=(width,height):raise ValueError("source-frame mismatch")
        crop,frame=crop_around_point(image.convert("RGB"),point,size=ZOOM_SIZE)
    target=Path(target)
    if target.exists():raise ValueError("create-only private zoom screenshot")
    target.parent.mkdir(parents=True,exist_ok=True)
    crop.save(target,"PNG")
    rw,rh=crop.size
    # Reuse original H61 intent/ordering only. Do not access bbox labels.
    prompt=build_prompt({"image_width":rw,"image_height":rh},original["task"])
    prompt+=(
        "\nThis is a ZOOMED CROP of the source GUI screenshot, NOT the full "
        "screen. Focus only on finding the "+endpoint+
        " endpoint of the drag instruction visible in this crop. "
        "Use LOCAL crop-image pixel coordinates in the exact JSON drag schema. "
        "The other drag endpoint may not appear; only your "+endpoint+
        "_point will be used to refine an earlier model prediction. "
        "Never infer absolute screen coordinates from the crop.")
    focus=dict(original)
    focus["image_path"]=str(target)
    focus["image_sha256"]=sha256_file(target)
    focus["image_width"],focus["image_height"]=rw,rh
    focus["prompt"]=prompt
    return Focus(focus,frame,endpoint)


def merge_refinements(original:dict,coarse_raw:str,
                      start_focus:Focus,start_raw:str,
                      end_focus:Focus,end_raw:str):
    """Fail closed when original coarse JSON invalid, else fallback per endpoint."""
    sw,sh=original["image_width"],original["image_height"]
    coarse=parse_action(coarse_raw,sw,sh)
    if coarse is None:raise ValueError("first-pass action failed native H61 parse")
    if start_focus.endpoint!="start" or end_focus.endpoint!="end":
        raise ValueError("start/end focus outputs must not be swapped")
    final={k:(list(v) if k.endswith("_point") else v) for k,v in coarse.items()}
    used={"start":False,"end":False}
    for endpoint,focus,raw in (("start",start_focus,start_raw),
                                ("end",end_focus,end_raw)):
        fw,fh=focus.scene["image_width"],focus.scene["image_height"]
        zoom_action=parse_action(raw,fw,fh)
        if zoom_action is None:continue
        pixel=zoom_action[endpoint+"_point"]
        corrected=focus.frame.zoomed_to_source(pixel)
        if not (0<=corrected[0]<sw and 0<=corrected[1]<sh):
            raise ValueError("zoom inverse point escaped original screenshot")
        final[endpoint+"_point"]=corrected
        used[endpoint]=True
    return json.dumps(final,sort_keys=True,separators=(",",":")),used

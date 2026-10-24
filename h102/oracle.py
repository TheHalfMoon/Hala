"""H102 strict source/frame/TRAIN geometry oracle; no training or inference."""
import hashlib
import json
from pathlib import Path

from h61.official_eval import build_prompt, parse_action, score
from h61.prompt_contract import drag_prompt, source_body

FIELDS = {"type","start_point","end_point","coordinate_space","ordered"}

def center(box):
    if type(box) is not list or len(box)!=4 or any(type(x) is not int for x in box):
        raise ValueError("four integer box edges required")
    return [(box[0]+box[2])//2,(box[1]+box[3])//2]

def in_bounds(box,w,h):
    return (type(w) is int and type(h) is int and w>0 and h>0
            and type(box) is list and len(box)==4
            and all(type(v) is int for v in box)
            and 0<=box[0]<=box[2]<w and 0<=box[1]<=box[3]<h)

def invalid_bbox_reasons(box,w,h):
    """Classify invalid source boundaries; no coordinates/identifiers exported."""
    if in_bounds(box,w,h):
        return ()
    if (type(box) is not list or len(box)!=4 or
            any(type(v) is not int for v in box)):
        return ("noninteger_or_malformed",)
    failures=[]
    if box[0]<0 or box[1]<0:
        failures.append("negative_left_or_top")
    if box[0]>box[2] or box[1]>box[3]:
        failures.append("inverted_box")
    if box[2]==w:
        failures.append("right_equals_width_half_open_candidate")
    elif box[2]>w:
        failures.append("right_exceeds_width")
    if box[3]==h:
        failures.append("bottom_equals_height_half_open_candidate")
    elif box[3]>h:
        failures.append("bottom_exceeds_height")
    if box[2]<0 or box[3]<0:
        failures.append("negative_right_or_bottom")
    return tuple(failures or ("invalid_dimensions_or_unknown_boundary",))


def inside(point,box):
    return (type(point) is list and len(point)==2 and
            all(type(v) is int for v in point) and
            box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3])

def make_action(start,end,ordered):
    if type(ordered) is not bool:
        raise ValueError("source ordered flag must be boolean")
    return {"type":"drag","start_point":list(start),"end_point":list(end),
            "coordinate_space":"source_image_pixels","ordered":ordered}

def assert_oracle(scene,start,end):
    w,h=scene["image_width"],scene["image_height"]
    if not all(in_bounds(scene["task"][k],w,h) for k in ("start_bbox","end_bbox")):
        raise ValueError("source box extends outside actual image pixels")
    action=make_action(start,end,scene["task"]["ordered"])
    raw=json.dumps(action,separators=(",",":"),sort_keys=True)
    if parse_action(raw,w,h)!=action:
        raise ValueError("source reference fails strict JSON parse")
    result=score([scene],{scene["id"]:raw})
    if result["task_success"]!=1 or result["action_valid"]!=1:
        raise ValueError("source reference fails unchanged H61 scorer")
    return True

def audit_eval_pair(original,framed):
    """Recompute image/frame/box/prompt identities from original source bytes."""
    from PIL import Image
    from h61.source_export import frame_target
    if original["id"]!=framed["id"] or original["domain"]!=framed["domain"]:
        raise ValueError("eval pairing mismatched")
    raw=Path(original["image_path"])
    target=Path(framed["image_path"])
    if (hashlib.sha256(raw.read_bytes()).hexdigest()!=original["image_sha256"]
            or hashlib.sha256(target.read_bytes()).hexdigest()!=framed["image_sha256"]):
        raise ValueError("original/framed media SHA256 mismatch")
    with Image.open(raw) as im: origin=im.size
    with Image.open(target) as im: frame=im.size
    if (origin!=(original["image_width"],original["image_height"]) or
            frame!=(framed["image_width"],framed["image_height"]) or
            frame!=frame_target(*origin)[:2] or
            framed["original_image_sha256"]!=original["image_sha256"]):
        raise ValueError("source-to-framed image dimension or identity drift")
    if (framed["original_image_width"],framed["original_image_height"])!=origin:
        raise ValueError("source screenshot geometry metadata drift")
    if framed["task"]["ordered"] is not original["task"]["ordered"]:
        raise ValueError("ordered direction flag mutated")
    for box in ("start_bbox","end_bbox"):
        src=original["task"][box]
        if not in_bounds(src,*origin):
            raise ValueError("source bbox outside original screenshot")
        expect=[int(round(src[i]*(frame[i%2]/origin[i%2]))) for i in range(4)]
        if framed["task"][box]!=expect or not in_bounds(expect,*frame):
            raise ValueError("box transform does not match actual image scaling")
    if framed["prompt"]!=build_prompt(framed,framed["task"]):
        raise ValueError("eval prompt contract source drift")
    start,end=(center(framed["task"][k]) for k in ("start_bbox","end_bbox"))
    if start==end:
        raise ValueError("degenerate eval source-target action")
    return assert_oracle(framed,start,end)

def audit_train_pair(authored,source,root):
    """Original H71 TRAIN labels remain exactly inside their source boxes."""
    from PIL import Image
    w,h=source["image_width"],source["image_height"]
    if (source["split"]!="train" or authored["split"]!="train"
            or source["action_type"]!="drag" or authored["id"]!=source["id"]
            or authored["image_path"]!=source["image_path"]
            or authored["image_sha256"]!=source["image_sha256"]
            or type(source["ordered"]) is not bool):
        raise ValueError("H71 derived/source TRAIN mismatch")
    src=Path(root)/authored["image_path"]
    if not src.is_file() or src.is_symlink():
        raise ValueError("H71 media unavailable or linked")
    if hashlib.sha256(src.read_bytes()).hexdigest()!=authored["image_sha256"]:
        raise ValueError("original TRAIN media SHA mismatch")
    with Image.open(src) as im:
        if im.size!=(w,h):
            raise ValueError("TRAIN image resolution does not equal label frame")
        if im.format!=authored["image_format"]:
            raise ValueError("TRAIN image format drift")
    if authored["prompt"]!=drag_prompt(source_body(source["intent"],source["ordered"]),
                                      w,h,source["ordered"]):
        raise ValueError("TRAIN prompt not source-bound")
    action=json.loads(authored["completion"])
    if (not isinstance(action,dict) or set(action)!=FIELDS or
            action!=source["completion"] or action["ordered"] is not source["ordered"]):
        raise ValueError("TRAIN JSON completion changed from original source")
    for point,box in (("start_point","start_bbox"),("end_point","end_bbox")):
        if not in_bounds(source[box],w,h) or not inside(action[point],source[box]):
            raise ValueError("TRAIN target coordinate outside source box/image")
    if action["start_point"]==action["end_point"]:
        raise ValueError("TRAIN degenerate drag target")
    scene={"id":source["id"],"domain":source["domain"],
           "image_width":w,"image_height":h,
           "task":{"ordered":source["ordered"],"start_bbox":source["start_bbox"],
                   "end_bbox":source["end_bbox"]}}
    assert_oracle(scene,action["start_point"],action["end_point"])
    return {"image_sha256":authored["image_sha256"],
            "language":authored["language"],"family":authored["family"]}

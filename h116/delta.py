"""H116 original TRAIN-only START-conditioned signed END displacement supervision.

No evaluation labels used, no trained model or optimizer allocated. Preserve
H113 original verified source points (not center-invented) and exact screenshots.
"""
from __future__ import annotations
import hashlib,json
from dataclasses import dataclass

from h113.dual_point import ROLES,COORDINATE_SPACE,relative_to_source_pixel
from h114.schedule import QualifiedSchedule
from h61.official_eval import strict_json

SCHEMA="hala.h116.original-train-exact-points-start-conditioned-displacement.v1"
SPACE="normalized_displacement_1000"
DELTA_ROLE="end_delta"

def validate_xy(value):
    if (type(value) is not list or len(value)!=2
            or any(type(v) is not int or not 0<=v<=1000 for v in value)):
        raise ValueError("authentic H113 normalized source pixel point required")
    return value


def prompt_end_delta(intent:str,anchor:list[int])->str:
    anchor=validate_xy(anchor)
    if (type(intent) is not str or not 4<=len(intent)<=1200
            or "bbox" in intent.lower()):
        raise ValueError("original bounded drag task text required; no answer labels")
    return (
        f"{intent}\n"
        "The ORIGINAL full screenshot is unchanged. Given the proposed "
        f"START anchor (normalized Qwen coordinates) [{anchor[0]},{anchor[1]}], "
        "locate the requested drag DESTINATION. "
        "Predict the SIGNED movement from the given START anchor to the END "
        "destination in normalized original-image units: dx=rightward, "
        "dy=downward, each signed integer in [-1000,1000]. "
        "The given START anchor is input context, not the destination. "
        "Return ONLY a JSON object with EXACT keys role, delta, coordinate_space, "
        f'role="{DELTA_ROLE}", coordinate_space="{SPACE}", '
        'delta=[signed_integer_dx,signed_integer_dy]. '
        "Do not output an absolute point and do not take any actions."
    )


def derive_pair(start:dict,end:dict)->tuple[dict,dict]:
    if (type(start) is not dict or type(end) is not dict
            or start.get("endpoint_role")!="start"
            or end.get("endpoint_role")!="end"
            or start.get("original_source_id")!=end.get("original_source_id")
            or not start.get("source_bound_validation_passed")
            or not end.get("source_bound_validation_passed")
            or start.get("split")!="train" or end.get("split")!="train"
            or not start.get("no_eval_or_test_source")
            or not end.get("no_eval_or_test_source")):
        raise ValueError("H116 authentic original H113 paired TRAIN-only source needed")
    for k in ("image_sha256","image_path","family","language",
              "source_image_width","source_image_height"):
        if start.get(k)!=end.get(k):
            raise ValueError("H116 START and END must share exact original source media")
    a=strict_json(start["completion"])
    b=strict_json(end["completion"])
    if (set(a)!={"role","point","coordinate_space"}
            or set(b)!={"role","point","coordinate_space"}
            or a["role"]!="start" or b["role"]!="end"
            or a["coordinate_space"]!=COORDINATE_SPACE
            or b["coordinate_space"]!=COORDINATE_SPACE):
        raise ValueError("only genuine H113 two distinct normalized original points")
    st=validate_xy(a["point"]);dst=validate_xy(b["point"])
    w,h=start["source_image_width"],start["source_image_height"]
    if (type(w) is not int or type(h) is not int
            or not 64<=w<=1024 or not 64<=h<=640):
        raise ValueError("incorrect original screenshot dimensions")
    relative_to_source_pixel(st,w,h)
    relative_to_source_pixel(dst,w,h)
    # Unchanged source task text is the prefix preceding the exact H113 contract.
    ssplit=start["prompt"].rsplit("\nLocate ONLY the START endpoint",1)
    esplit=end["prompt"].rsplit("\nLocate ONLY the END endpoint",1)
    if (len(ssplit)!=2 or len(esplit)!=2 or ssplit[0]!=esplit[0]
            or not ssplit[1].startswith(" of the requested drag ")
            or not esplit[1].startswith(" of the requested drag ")):
        raise ValueError("H116 original H113 start/end intent differs")
    instruction=ssplit[0]
    displacement=[dst[i]-st[i] for i in range(2)]
    if any(abs(n)>1000 for n in displacement) or [st[i]+displacement[i] for i in range(2)]!=dst:
        raise ValueError("signed delta cannot recreate exact ORIGINAL END label")
    derived=dict(end)
    derived.update({
       "schema":SCHEMA,
       "id":end["original_source_id"]+"::end_delta",
       "endpoint_role":DELTA_ROLE,
       "prompt":prompt_end_delta(instruction,st),
       "completion":json.dumps(
           {"role":DELTA_ROLE,"delta":displacement,"coordinate_space":SPACE},
           sort_keys=True,separators=(",",":")),
    })
    # START source remains EXACT H113 and END media/provenance unchanged.
    return dict(start),derived


@dataclass(frozen=True)
class DeltaSchedule:
    selected:tuple[dict,...]
    summary:dict


def derive_schedule(parent:QualifiedSchedule)->DeltaSchedule:
    if not isinstance(parent,QualifiedSchedule):
        raise ValueError("qualified original H114 parent required")
    if (parent.summary.get("source_split")!="TRAIN_ONLY"
            or parent.summary.get("pilot_selected_original_screenshots")!=128
            or parent.summary.get("selected_dual_point_optimizer_steps")!=256
            or len(parent.selected)!=256):
        raise ValueError("H116 only original owner-verified first H105 128 pairs")
    mixed=[]
    deltas=[]
    for i in range(0,len(parent.selected),2):
        first,second=derive_pair(parent.selected[i],parent.selected[i+1])
        mixed.extend([first,second])
        values=strict_json(second["completion"])["delta"]
        deltas.append(tuple(values))
    unique=len(set(z["image_sha256"] for z in mixed))
    if (len(mixed)!=256 or unique!=128
            or any(x["endpoint_role"]!="start" or y["endpoint_role"]!=DELTA_ROLE
                   for x,y in zip(mixed[::2],mixed[1::2]))
            or any(all(v==0 for v in delta) for delta in deltas)):
        raise ValueError("H116 source point pairs missing or collapsed")
    data=b"".join(json.dumps(z,ensure_ascii=False,sort_keys=True,
                            separators=(",",":")).encode("utf8")+b"\n" for z in mixed)
    summary={
       "schema":SCHEMA,"source_split":"TRAIN_ONLY",
       "source_images":128,"start_point_roles":128,
       "conditioned_signed_destination_roles":128,"optimizer_candidate_steps":256,
       "all_original_screenshots_unchanged":True,
       "end_relative_delta_signed_nonzero":True,
       "original_h113_parent_sha256":parent.summary["original_h113_point_jsonl_sha256"],
       "original_h105_source_selection_sha256":parent.summary["original_h105_frozen_plan_sha256"],
       "original_languages":parent.summary["selected_languages"],
       "original_families":parent.summary["selected_families"],
       "derived_private_training_jsonl_sha256":hashlib.sha256(data).hexdigest(),
       "individual_source_ids_and_coordinates_in_receipt":False,
       "no_EVAL_or_TEST_in_training":True,
       "actual_GPU_training_executed":False}
    return DeltaSchedule(tuple(mixed),summary)


def compose_model_drag_from_delta(original:dict,start_raw:str,delta_raw:str)->str|None:
    """Original H61 scorer bridge, using ONLY two model-generated values.

    No GT start/end target boxes enter prompts or math. All malformed or
    source-out-of-bounds endpoints fail closed, no clipping or repair.
    """
    from h61.official_eval import parse_action
    try:
        a=strict_json(start_raw)
        b=strict_json(delta_raw)
    except (ValueError,TypeError):
        return None
    if (type(a) is not dict or set(a)!={"role","point","coordinate_space"}
            or a.get("role")!="start" or a.get("coordinate_space")!=COORDINATE_SPACE
            or type(b) is not dict or set(b)!={"role","delta","coordinate_space"}
            or b.get("role")!=DELTA_ROLE or b.get("coordinate_space")!=SPACE):
        return None
    try:
        st=validate_xy(a["point"])
        dx=b["delta"]
        if (type(dx) is not list or len(dx)!=2
                or any(type(n) is not int or not -1000<=n<=1000 for n in dx)):
            return None
        target=[st[i]+dx[i] for i in range(2)]
        target=validate_xy(target)
        w,h=original["image_width"],original["image_height"]
        first=relative_to_source_pixel(st,w,h)
        second=relative_to_source_pixel(target,w,h)
        output={"type":"drag","coordinate_space":"source_image_pixels",
                "ordered":original["task"]["ordered"],
                "start_point":first,"end_point":second}
        serialized=json.dumps(output,sort_keys=True,separators=(",",":"))
        return serialized if parse_action(serialized,w,h)==output else None
    except (KeyError,TypeError,ValueError):
        return None

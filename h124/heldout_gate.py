"""Predeclared H124 original TRAIN-only START+END full drag strict gate.

Freeze BEFORE any H124 GPU output. No original EVAL/TEST or hidden
labels. False/failed results are recorded without retuning thresholds.
"""
from collections import Counter
import hashlib
from h61.official_eval import strict_json
from h124.objective import parse
from h124.schedule import Schedule,HOLDOUT
from h124.training_plan import get_plan,TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA
from h123.schedule import FAMILIES

SCHEMA="hala.h124.original-TRAIN128-joint-START-END-full-drag-gate.v1"
MIN_NATIVE_VALID=112
MIN_START_HITS=32
MIN_END_HITS=32
MIN_FULL_DRAG_HITS=16
MIN_FULL_EACH_FAMILY=2
MIN_UNIQUE_START=80
MIN_UNIQUE_END=80
MAX_MODAL_FRACTION=0.125


def inside(point,box):
    return box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]


def score(cases,responses,schedule:Schedule):
    if (type(cases) is not list or len(cases)!=HOLDOUT
            or type(responses) is not list or len(responses)!=HOLDOUT):
        raise ValueError("H124 128 genuine original TRAIN heldout paired neural calls needed")
    get_plan(schedule)
    if (schedule.summary.get("train_private_sha256")!=TRAIN_SHA
            or schedule.summary.get("heldout_private_sha256")!=HELDOUT_SHA
            or schedule.summary.get("schedule_sha256")!=SCHEDULE_SHA):
        raise ValueError("H124 model heldout source SHA schedule differs")
    seen=set();families=Counter();hits=Counter()
    start_pts=Counter();end_pts=Counter()
    native=start=ending=full=0
    for case,model,role in zip(cases,responses,schedule.heldout):
        if (not isinstance(case,dict) or not isinstance(model,dict)
                or case.get("split")!="train"
                or case.get("source_split")!="TRAIN_ONLY_HELDOUT"
                or case.get("id")!=role.get("original_source_id")
                or case.get("image_sha256")!=role.get("image_sha256")
                or case.get("prompt")!=role.get("prompt")
                or case.get("family")!=role.get("family")
                or case.get("family") not in FAMILIES
                or case.get("language")!="en"
                or case["id"] in seen or model.get("id")!=case["id"]
                or model.get("image_sha256")!=case["image_sha256"]
                or model.get("prompt_sha256")!=hashlib.sha256(
                    case["prompt"].encode("utf8")).hexdigest()):
            raise ValueError("H124 exact original image/prompt/heldout model identity drift")
        seen.add(case["id"])
        raw=model.get("raw_output")
        if (type(raw) is not str or model.get("output_sha256")!=
                hashlib.sha256(raw.encode("utf8")).hexdigest()):
            raise ValueError("H124 unmodified original neural raw response SHA missing")
        w,h=case.get("image_width"),case.get("image_height")
        if (type(w) is not int or type(h) is not int
                or not 64<=w<=1024 or not 64<=h<=640):
            raise ValueError("H124 original image dimensions absent")
        for name in ("start_bbox","end_bbox"):
            box=case.get(name)
            if (type(box) is not list or len(box)!=4 or
                    any(type(x) is not int for x in box)
                    or not 0<=box[0]<=box[2]<w
                    or not 0<=box[1]<=box[3]<h):
                raise ValueError("H124 real source START/END boxes absent")
        families[case["family"]]+=1
        points=parse(raw,w,h)
        if points is None:
            continue
        native+=1
        original=strict_json(raw)
        start_pts[tuple(original["start_point"])]+=1
        end_pts[tuple(original["end_point"])]+=1
        s=inside(points[0],case["start_bbox"])
        e=inside(points[1],case["end_bbox"])
        if s:start+=1
        if e:ending+=1
        if s and e:
            full+=1
            hits[case["family"]]+=1
    if set(families)!=set(FAMILIES) or any(families[f]!=32 for f in FAMILIES):
        raise ValueError("H124 four distinct complex original TRAIN heldout domains required")
    modal_start=max(start_pts.values())/native if native else 1.
    modal_end=max(end_pts.values())/native if native else 1.
    passed=(native>=MIN_NATIVE_VALID and start>=MIN_START_HITS
            and ending>=MIN_END_HITS and full>=MIN_FULL_DRAG_HITS
            and all(hits[f]>=MIN_FULL_EACH_FAMILY for f in FAMILIES)
            and len(start_pts)>=MIN_UNIQUE_START
            and len(end_pts)>=MIN_UNIQUE_END
            and max(modal_start,modal_end)<=MAX_MODAL_FRACTION)
    return {
      "schema":SCHEMA,
      "status":"PASS_H124_ORIGINAL_TRAIN128_FULL_DRAG_GATE" if passed
               else "FAIL_H124_ORIGINAL_TRAIN128_FULL_DRAG_GATE",
      "actual_model_inference_calls":HOLDOUT,
      "native_valid_joint_geometry":native,
      "strict_original_START_hits":start,
      "strict_original_END_hits":ending,
      "strict_original_full_drag_hits":full,
      "full_drag_hits_per_family":dict(sorted((f,hits[f]) for f in FAMILIES)),
      "heldout_family_counts":dict(sorted(families.items())),
      "unique_native_START_points":len(start_pts),
      "unique_native_END_points":len(end_pts),
      "modal_START_fraction":round(modal_start,5),
      "modal_END_fraction":round(modal_end,5),
      "thresholds":{
         "min_native_valid":MIN_NATIVE_VALID,
         "min_START_hits":MIN_START_HITS,
         "min_END_hits":MIN_END_HITS,
         "min_full_drag_hits":MIN_FULL_DRAG_HITS,
         "min_full_drag_each_complex_family":MIN_FULL_EACH_FAMILY,
         "min_unique_START_points":MIN_UNIQUE_START,
         "min_unique_END_points":MIN_UNIQUE_END,
         "max_modal_fraction":MAX_MODAL_FRACTION},
      "heldout_gate_passed":passed,
      "source_schedule_sha256":SCHEDULE_SHA,
      "no_original_EVAL_TEST":True,
      "no_prior_H113_H123_source_reused":True,
      "external_GUI_skill_qualified":False}

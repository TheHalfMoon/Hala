"""Pre-registered H123 *original TRAIN-heldout only* END qualification.

Freeze this rubric BEFORE any H123 model or its 128 heldout outputs exist.
No original public EVAL/TEST data, no training/checkpoint selection.
"""
from collections import Counter
import hashlib

from h61.official_eval import strict_json
from h123.objective import parse
from h123.schedule import FAMILIES,HOLDOUT,Schedule
from h123.training_plan import (get_plan,SCHEDULE_SHA,
                                TRAIN_SHA,HELDOUT_SHA)

SCHEMA="hala.h123.original-complex-TRAIN128-END-gate.v1"
MIN_VALID=112
MIN_END_HITS=32
MIN_EACH_FAMILY_HITS=4
MIN_UNIQUE_POINT=80
MAX_MODAL_FRACTION=0.125


def score(cases,model_responses,schedule:Schedule):
    if (type(cases) is not list or len(cases)!=HOLDOUT
            or type(model_responses) is not list
            or len(model_responses)!=HOLDOUT):
        raise ValueError("H123 128 paired original TRAIN-heldout neural evidence needed")
    get_plan(schedule)
    if (schedule.summary.get("schedule_sha256")!=SCHEDULE_SHA
            or schedule.summary.get("train_private_sha256")!=TRAIN_SHA
            or schedule.summary.get("heldout_private_sha256")!=HELDOUT_SHA):
        raise ValueError("H123 original 896/128 source schedule SHA changed")
    seen=set();families=Counter();family_hits=Counter()
    points=Counter();valid=hit=0
    for case,obs,ref in zip(cases,model_responses,schedule.heldout):
        if (type(case) is not dict or type(obs) is not dict
                or case.get("split")!="train"
                or case.get("source_split")!="TRAIN_ONLY_HELDOUT"
                or case.get("id")!=ref.get("original_source_id")
                or case.get("image_sha256")!=ref.get("image_sha256")
                or case.get("prompt")!=ref.get("prompt")
                or case.get("family")!=ref.get("family")
                or case.get("family") not in FAMILIES
                or case.get("language")!="en"
                or case.get("id") in seen
                or obs.get("id")!=case.get("id")
                or obs.get("image_sha256")!=case.get("image_sha256")
                or obs.get("prompt_sha256")!=hashlib.sha256(
                    case["prompt"].encode("utf8")).hexdigest()):
            raise ValueError("H123 original private source/prompt/inference IDs drift")
        seen.add(case["id"])
        raw=obs.get("raw_output")
        if (type(raw) is not str or
                obs.get("output_sha256")!=hashlib.sha256(raw.encode("utf8")).hexdigest()):
            raise ValueError("H123 real neural raw response SHA mismatch")
        w,h,box=case.get("image_width"),case.get("image_height"),case.get("end_bbox")
        if (type(w) is not int or type(h) is not int or not 64<=w<=1024
                or not 64<=h<=640 or type(box) is not list or len(box)!=4
                or any(type(v) is not int for v in box)
                or not 0<=box[0]<=box[2]<w or not 0<=box[1]<=box[3]<h):
            raise ValueError("H123 genuine original source END target box required")
        families[case["family"]]+=1
        pixel=parse(raw,w,h)
        if pixel is None:
            continue
        valid+=1
        pt=strict_json(raw)["point"]
        points[tuple(pt)]+=1
        if box[0]<=pixel[0]<=box[2] and box[1]<=pixel[1]<=box[3]:
            hit+=1
            family_hits[case["family"]]+=1
    if set(families)!=set(FAMILIES) or any(families[f]!=32 for f in FAMILIES):
        raise ValueError("H123 exactly 32 unique original heldout images per complex domain")
    modal=max(points.values())/valid if valid else 1.0
    qualified=(valid>=MIN_VALID and hit>=MIN_END_HITS
               and len(points)>=MIN_UNIQUE_POINT and modal<=MAX_MODAL_FRACTION
               and all(family_hits[f]>=MIN_EACH_FAMILY_HITS for f in FAMILIES))
    return {
        "schema":SCHEMA,"status":"PASS_H123_INTERNAL_TRAIN128_END_GATE"
                     if qualified else "FAIL_H123_INTERNAL_TRAIN128_END_GATE",
        "source_split":"TRAIN_ONLY_HELDOUT",
        "model_inference_calls":HOLDOUT,
        "strict_valid_END_dualgrid_and_point":valid,
        "strict_END_hits":hit,
        "hits_per_family":dict(sorted((f,family_hits[f]) for f in FAMILIES)),
        "source_family_counts":dict(sorted(families.items())),
        "unique_native_normalized_END_points":len(points),
        "modal_normalized_point_fraction":round(modal,5),
        "thresholds":{
            "min_native_valid":MIN_VALID,
            "min_original_END_hits":MIN_END_HITS,
            "min_hits_each_complex_family":MIN_EACH_FAMILY_HITS,
            "min_unique_native_points":MIN_UNIQUE_POINT,
            "max_single_point_fraction":MAX_MODAL_FRACTION},
        "heldout_gate_passed":qualified,
        "schedule_sha256":SCHEDULE_SHA,
        "original_PUBLIC_EVAL_or_hidden_TEST_read":False,
        "no_previous_H113_H122_source_reused":True,
        "not_qualified_for_external_GUI_skill":True,
    }

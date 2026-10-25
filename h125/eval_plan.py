"""H125 frozen 64 new original TRAIN-only GUI 2B paired grounding gate."""
from collections import Counter
import hashlib
from h125.model_contracts import MODELS,parse_point
from h125.source import Schedule,TOTAL,PER_FAMILY,FAMILIES

SCHEMA="hala.h125.original-train64-native-pretrained-two2b-v1"
SOURCE_SHA="d95392f22d40b2a9c34a94421a790c94c5107e7bccff4b023df605b239c5ce49"
SCHEDULE_SHA="81961961be80527d241850f6deb14ce9c9511833c85f58f2b3430b3d74eb6a48"
MIN_VALID=48
MIN_START=12
MIN_END=12
MIN_FULL=8
MIN_FULL_PER_FAMILY=1
MIN_UNIQUE=32
MAX_MODAL=0.20


def get_plan(plan:Schedule):
    if (not isinstance(plan,Schedule) or len(plan.cases)!=TOTAL
            or len(plan.indices)!=TOTAL or len(set(plan.indices))!=TOTAL
            or plan.summary.get("schedule_sha256")!=SCHEDULE_SHA
            or plan.summary.get("private_selected_source_sha256")!=SOURCE_SHA
            or plan.summary.get("prior_original_screenshots_excluded")!=2560
            or plan.summary.get("no_prior_heldout_reuse") is not True
            or plan.summary.get("original_EVAL_or_TEST_used") is not False):
        raise ValueError("H125 exact SHA frozen new original TRAIN64 source required")
    return plan


def _inside(point,box):
    return box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]


def score(cases,observations,source:Schedule,model:str):
    get_plan(source)
    if model not in MODELS or type(cases) is not list or type(observations) is not list:
        raise ValueError("H125 exact two registered pretrained models/real source required")
    if len(cases)!=len(observations) or len(cases)!=TOTAL:
        raise ValueError("H125 exact original 64 paired native GUI grounding responses")
    known=set();families=Counter();full_family=Counter()
    starts=Counter();ends=Counter();valid=start=end=full=0
    for case,obs,role in zip(cases,observations,source.cases):
        if (type(case) is not dict or type(obs) is not dict
                or case.get("split")!="train"
                or case.get("source_split")!="TRAIN_ONLY_FRESH_NEVER_TRAINED"
                or case.get("id")!=role["id"]
                or case.get("family")!=role["family"]
                or case.get("image_sha256")!=role["image_sha256"]
                or case.get("id") in known
                or case.get("family") not in FAMILIES
                or obs.get("id")!=case.get("id")
                or obs.get("image_sha256")!=case.get("image_sha256")
                or obs.get("arm")!=model):
            raise ValueError("H125 original source model response pairing changed")
        known.add(case["id"]);families[case["family"]]+=1
        for name in ("start","end"):
            prompt=case.get("prompt_"+name)
            raw=obs.get("raw_"+name)
            if (type(prompt) is not str or type(raw) is not str
                    or obs.get("prompt_"+name+"_sha256")!=hashlib.sha256(
                        prompt.encode("utf8")).hexdigest()
                    or obs.get("raw_"+name+"_sha256")!=hashlib.sha256(
                        raw.encode("utf8")).hexdigest()):
                raise ValueError("H125 raw per-endpoint pretrained output/prompt SHA changed")
        w,h=case.get("image_width"),case.get("image_height")
        if type(w) is not int or type(h) is not int or not 64<=w<=1024 or not 64<=h<=640:
            raise ValueError("H125 genuine original source frame dimensions required")
        for name in ("start","end"):
            b=case.get(name+"_bbox")
            if (type(b) is not list or len(b)!=4 or
                    any(type(x) is not int for x in b)
                    or not 0<=b[0]<=b[2]<w or not 0<=b[1]<=b[3]<h):
                raise ValueError("H125 genuine source START END bounding box invalid")
        predicted_start=parse_point(model,obs["raw_start"],w,h)
        predicted_end=parse_point(model,obs["raw_end"],w,h)
        if predicted_start is None or predicted_end is None:
            continue
        valid+=1
        starts[predicted_start]+=1;ends[predicted_end]+=1
        sh=_inside(predicted_start,case["start_bbox"])
        eh=_inside(predicted_end,case["end_bbox"])
        start+=sh;end+=eh
        if sh and eh:
            full+=1;full_family[case["family"]]+=1
    if set(families)!=set(FAMILIES) or any(families[f]!=PER_FAMILY for f in FAMILIES):
        raise ValueError("H125 exactly 16 authentic new GUI sources per domain")
    ms=max(starts.values())/valid if valid else 1.0
    me=max(ends.values())/valid if valid else 1.0
    passed=(valid>=MIN_VALID and start>=MIN_START and end>=MIN_END
            and full>=MIN_FULL
            and all(full_family[f]>=MIN_FULL_PER_FAMILY for f in FAMILIES)
            and len(starts)>=MIN_UNIQUE and len(ends)>=MIN_UNIQUE
            and max(ms,me)<=MAX_MODAL)
    return {
      "schema":SCHEMA,"model_arm":model,
      "status":"PASS_H125_INTERNAL_PRETRAINED_GUI64_GATE"
                if passed else "FAIL_H125_INTERNAL_PRETRAINED_GUI64_GATE",
      "original_new64_neural_pairs":TOTAL,
      "native_valid_both_points":valid,
      "strict_original_START_hits":start,
      "strict_original_END_hits":end,
      "strict_original_full_drag_hits":full,
      "full_drag_hits_per_family":dict(sorted((f,full_family[f]) for f in FAMILIES)),
      "source_family_counts":dict(sorted(families.items())),
      "unique_START_pixels":len(starts),"unique_END_pixels":len(ends),
      "modal_START":round(ms,5),"modal_END":round(me,5),
      "source_schedule_sha256":SCHEDULE_SHA,
      "native_output_space":MODELS[model].coordinate_space,
      "frozen_thresholds":{
          "min_native_valid":MIN_VALID,"min_START_hits":MIN_START,
          "min_END_hits":MIN_END,"min_FULL_hits":MIN_FULL,
          "min_FULL_per_family":MIN_FULL_PER_FAMILY,
          "min_unique_both":MIN_UNIQUE,"max_modal":MAX_MODAL},
      "internal_gate_passed":passed,
      "no_original_EVAL_TEST":True,
      "no_training_on_old_heldout":True,
      "no_model_finetune_or_optimizer":True,
      "external_GUI_skill_qualified":False}

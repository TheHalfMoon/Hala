"""H122 pre-registered original TRAIN-only, optimizer-disjoint 64 END check.

No model fitting, checkpoint selection or official EVAL/TEST reading here.
Only owner-private H113 original TRAIN holdout source+neural bytes are scored.
"""
import hashlib
from collections import Counter
from h122.grid import parse_end
from h122.schedule import HOLDOUT

SCHEMA="hala.h122.owner-private-original-train64-heldout-END-eval.v1"
FROZEN_SPLIT_SHA="f986eeb5ca0c242a1d000ebe4474dc429abdc54514b20dd9e82e1537cbd57ba2"
MIN_END_HITS=8
MIN_VALID_ACTIONS=48
MIN_UNIQUE_VALID_POINTS=24
MAX_MODAL_POINT_FRACTION=0.25
MIN_FAMILIES_WITH_HIT=3

def qualify_prediction_evidence(cases,raw_evidence):
    """Verify paired screenshot/prompt/raw-response SHAs before ANY bbox score."""
    if type(cases) is not list or len(cases)!=HOLDOUT or type(raw_evidence) is not list or len(raw_evidence)!=HOLDOUT:
        raise ValueError("H122 exactly 64 original heldout TRAIN inference calls required")
    seen=set()
    outputs={}
    for item,observation in zip(cases,raw_evidence):
        if type(item) is not dict or type(observation) is not dict:
            raise ValueError("H122 exact original private evidence objects")
        source_id=item.get("id")
        if (type(source_id) is not str or not source_id
                or source_id in seen or item.get("split")!="train"
                or item.get("source_split")!="TRAIN_ONLY_HELDOUT"
                or type(item.get("family")) is not str or not item["family"]
                or type(item.get("image_sha256")) is not str
                or len(item["image_sha256"])!=64
                or type(item.get("prompt")) is not str):
            raise ValueError("H122 source images not heldout original TRAIN")
        if (observation.get("id")!=source_id or
                observation.get("image_sha256")!=item["image_sha256"] or
                observation.get("prompt_sha256")!=hashlib.sha256(item["prompt"].encode("utf8")).hexdigest()):
            raise ValueError("H122 original image/prompt SHA differs from actual model inference")
        raw=observation.get("raw_output")
        if type(raw) is not str or hashlib.sha256(raw.encode("utf8")).hexdigest()!=observation.get("output_sha256"):
            raise ValueError("H122 neural output byte SHA changed")
        seen.add(source_id)
        outputs[source_id]=raw
    return outputs

def score_heldout(cases,raw_evidence,split_sha=FROZEN_SPLIT_SHA):
    """Score genuine target boxes ONLY AFTER verifying model responses.

    Strict gate does not establish external general GUI skill.
    """
    if split_sha!=FROZEN_SPLIT_SHA:
        raise ValueError("H122 fixed original 448/64 split SHA differs")
    raw=qualify_prediction_evidence(cases,raw_evidence)
    valid=0
    end_hit=0
    per_family=Counter()
    family_hits=Counter()
    points=Counter()
    grid_cells=Counter()
    for item in cases:
        w,h=item.get("image_width"),item.get("image_height")
        box=item.get("end_bbox")
        if (type(w) is not int or type(h) is not int or
                not 64<=w<=1024 or not 64<=h<=640 or
                type(box) is not list or len(box)!=4 or
                any(type(c) is not int for c in box) or
                not 0<=box[0]<=box[2]<w or not 0<=box[1]<=box[3]<h):
            raise ValueError("H122 original TRAIN target box unreadable")
        fam=item["family"]
        per_family[fam]+=1
        end=parse_end(raw[item["id"]],w,h)
        if end is None:
            continue
        valid+=1
        points[tuple(end)]+=1
        grid_cells[(min(9,end[0]*10//w),min(9,end[1]*10//h))]+=1
        if box[0]<=end[0]<=box[2] and box[1]<=end[1]<=box[3]:
            end_hit+=1
            family_hits[fam]+=1
    if len(per_family)!=5 or sum(per_family.values())!=HOLDOUT:
        raise ValueError("H122 all five original TRAIN families required")
    modal=(max(points.values())/valid) if valid else 1.0
    gate=(valid>=MIN_VALID_ACTIONS and end_hit>=MIN_END_HITS and
          len(points)>=MIN_UNIQUE_VALID_POINTS and
          modal<=MAX_MODAL_POINT_FRACTION and
          len(family_hits)>=MIN_FAMILIES_WITH_HIT)
    return {
        "schema":SCHEMA,
        "status":"PASS_H122_INTERNAL_HELDOUT_END_GATE" if gate else "FAIL_H122_INTERNAL_HELDOUT_END_GATE",
        "source_split":"TRAIN_ONLY_HELDOUT",
        "original_heldout_scenes":HOLDOUT,
        "strict_valid_END_grid_and_point":valid,
        "strict_END_hits":end_hit,
        "unique_valid_END_points":len(points),
        "distinct_valid_grid_cells":len(grid_cells),
        "modal_point_fraction":round(modal,5),
        "original_train_families":dict(sorted(per_family.items())),
        "hits_per_family":dict(sorted(family_hits.items())),
        "thresholds":{"minimum_END_hits":MIN_END_HITS,
                      "minimum_valid":MIN_VALID_ACTIONS,
                      "minimum_unique_valid_points":MIN_UNIQUE_VALID_POINTS,
                      "maximum_modal_point_fraction":MAX_MODAL_POINT_FRACTION,
                      "minimum_families_with_hit":MIN_FAMILIES_WITH_HIT},
        "holdout_gate_passed":gate,
        "frozen_split_sha256":FROZEN_SPLIT_SHA,
        "all_model_raw_screenshot_and_prompt_SHA_verified":True,
        "no_original_public_EVAL_or_TEST_used":True,
        "no_holdout_images_used_for_optimizer":True,
        "GUI_skill_improvement_claimed":False,
        "external_evaluation_required_even_if_gate_passes":True,
    }

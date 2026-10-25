"""H121 private, aggregate-only geometric telemetry from TRAIN and model outputs.

Training distribution is admitted from SHA-locked original H113 TRAIN.
Predicted EVAL model outputs MAY be summarized, but NEVER used to derive
TRAIN targets, retune scored experiments, or create/modify checkpoint data.
No source IDs, labels or individual pixel/point values leave this module.
"""
from collections import Counter
from math import hypot
from statistics import median
from h61.official_eval import strict_json

SCHEMA="hala.h121.private-aggregate-geometric-diagnostics.v1"
SPACE="normalized_0_1000"
DELTA="normalized_displacement_1000"

def strict_xy(value, signed=False):
    low=-1000 if signed else 0
    if (type(value) is not list or len(value)!=2 or
            any(type(v) is not int or not low<=v<=1000 for v in value)):
        raise ValueError("strict bounded integer point or displacement required")
    return value

def strict_role(raw,role):
    if role not in ("start","end","end_delta"):
        raise ValueError("exact native source point or delta role required")
    if not isinstance(raw,str) or not 1<=len(raw)<=8192:
        return None
    try:
        obj=strict_json(raw)
    except (ValueError,TypeError):
        return None
    key="delta" if role=="end_delta" else "point"
    expected=DELTA if role=="end_delta" else SPACE
    if (type(obj) is not dict or set(obj)!={"role",key,"coordinate_space"}
            or obj["role"]!=role or obj["coordinate_space"]!=expected):
        return None
    try:
        return strict_xy(obj[key],signed=(role=="end_delta"))
    except ValueError:
        return None

def _distribution(vectors):
    if not isinstance(vectors,list) or any(
           type(v) is not list or len(v)!=2 or
           any(type(n) is not int or not -1000<=n<=1000 for n in v)
           for v in vectors):
        raise ValueError("bounded displacement vectors required")
    if not vectors:
        return {"count":0,"median_norm":None,"p90_norm":None,
                "near_zero_fraction":None,"positive_x_fraction":None,
                "positive_y_fraction":None,"direction_buckets":{}}
    magnitudes=sorted(hypot(v[0],v[1]) for v in vectors)
    n=len(vectors)
    directions=Counter(("right" if v[0]>0 else "left" if v[0]<0 else "zero_x")+
                       "/" +("down" if v[1]>0 else "up" if v[1]<0 else "zero_y")
                       for v in vectors)
    return {
        "count":n,
        "unique_displacement_vectors":len({tuple(v) for v in vectors}),
        "most_common_vector_fraction":round(max(Counter(tuple(v) for v in vectors).values())/n,4),
        "median_norm":round(median(magnitudes),2),
        "p90_norm":round(magnitudes[min(n-1,(9*n)//10)],2),
        "near_zero_fraction":round(sum(hypot(*v)<64 for v in vectors)/n,4),
        "positive_x_fraction":round(sum(v[0]>0 for v in vectors)/n,4),
        "positive_y_fraction":round(sum(v[1]>0 for v in vectors)/n,4),
        "direction_buckets":dict(sorted(directions.items())),
    }

def summarize_train_only_pairs(original_h113):
    """512 original verified TRAIN dual-point pairs, no raw source receipt."""
    from h114.schedule import QualifiedSchedule, ORIGINAL_H113_DERIVED_SHA
    if (not isinstance(original_h113,QualifiedSchedule)
            or original_h113.summary.get("source_split")!="TRAIN_ONLY"
            or original_h113.summary.get("original_h113_point_jsonl_sha256")!=ORIGINAL_H113_DERIVED_SHA
            or len(original_h113.selected)!=1024):
        raise ValueError("original SHA-bound H113 512 unique TRAIN pairs required")
    vectors=[]
    starts=[]
    source_images=set()
    for first,last in zip(original_h113.selected[::2],original_h113.selected[1::2]):
        if (first.get("original_source_id")!=last.get("original_source_id")
                or first.get("image_sha256")!=last.get("image_sha256")
                or first.get("split")!="train" or last.get("split")!="train"):
            raise ValueError("TRAIN original source/role identity mismatch")
        if first["image_sha256"] in source_images:
            raise ValueError("duplicate original TRAIN screenshot")
        source_images.add(first["image_sha256"])
        a=strict_role(first.get("completion"),"start")
        b=strict_role(last.get("completion"),"end")
        if a is None or b is None:
            raise ValueError("original TRAIN point role malformed")
        starts.append(a)
        vectors.append([b[0]-a[0],b[1]-a[1]])
    if len(vectors)!=512:
        raise ValueError("exact original 512 TRAIN images required")
    return {"schema":SCHEMA,"source_split":"TRAIN_ONLY",
            "unique_source_images":512,
            "source_SHA_bound":True,"true_train_displacement":_distribution(vectors),
            "no_EVAL_or_TEST_used_for_TRAIN":True,
            "no_raw_source_identifiers_or_coordinates_exposed":True}

def summarize_model_predictions(start_outputs,delta_outputs):
    """Prediction-only geometry, no target labels or benchmark scores.

    Input is raw neural response strings in original inference order; matched
    positions refer to model outputs only, NEVER original EVAL targets.
    """
    if (type(start_outputs) is not list or type(delta_outputs) is not list
            or len(start_outputs)!=len(delta_outputs)
            or not 1<=len(start_outputs)<=1024):
        raise ValueError("paired raw neural outputs required")
    vectors=[];bad_starts=0;bad_deltas=0;off_image=0
    start_values=[]
    for start_raw,delta_raw in zip(start_outputs,delta_outputs):
        start=strict_role(start_raw,"start")
        delta=strict_role(delta_raw,"end_delta")
        if start is None:
            bad_starts+=1
        else:
            start_values.append(start)
        if delta is None:
            bad_deltas+=1
        if start is None or delta is None:
            continue
        end=[start[i]+delta[i] for i in range(2)]
        if any(not 0<=n<=1000 for n in end):
            off_image+=1
        vectors.append(delta)
    return {"schema":SCHEMA,"prediction_only":True,
            "model_calls_first_start":len(start_outputs),
            "start_schema_valid":len(start_outputs)-bad_starts,
            "end_delta_schema_valid":len(start_outputs)-bad_deltas,
            "paired_parsable":len(vectors),
            "predicted_end_off_original_image":off_image,
            "predicted_signed_displacement":_distribution(vectors),
            "no_true_EVAL_locations_or_boxes_used":True,
            "no_raw_model_coordinates_or_outputs_exposed":True}



def summarize_model_absolute(start_outputs,end_outputs):
    """H114 predicted START and END only; never inspect EVAL labels."""
    if (type(start_outputs) is not list or type(end_outputs) is not list
            or len(start_outputs)!=len(end_outputs)
            or not 1<=len(start_outputs)<=1024):
        raise ValueError("matched absolute neural predictions required")
    vectors=[];invalid_start=invalid_end=0
    for a,b in zip(start_outputs,end_outputs):
        start=strict_role(a,"start")
        end=strict_role(b,"end")
        invalid_start+=start is None
        invalid_end+=end is None
        if start is not None and end is not None:
            vectors.append([end[0]-start[0],end[1]-start[1]])
    return {"schema":SCHEMA,"prediction_only":True,
            "model_calls_per_endpoint":len(start_outputs),
            "start_schema_valid":len(start_outputs)-invalid_start,
            "end_schema_valid":len(start_outputs)-invalid_end,
            "paired_parsable":len(vectors),
            "predicted_displacement":_distribution(vectors),
            "no_true_EVAL_locations_or_boxes_used":True,
            "no_raw_model_coordinates_or_outputs_exposed":True}



def training_vs_prediction_diagnostic(train_report,prediction_report):
    """Exploratory NO-LABEL guard. Never a model-quality verdict.

    Relative aggregate shifts may reflect different TRAIN/EVAL task mixes;
    this flags outputs worth investigating, not supervised correction or
    any checkpoint-selection rule. Thresholds are descriptive and post-hoc.
    """
    if (type(train_report) is not dict or type(prediction_report) is not dict
            or train_report.get("source_split")!="TRAIN_ONLY"
            or train_report.get("unique_source_images")!=512
            or prediction_report.get("prediction_only") is not True
            or prediction_report.get("no_true_EVAL_locations_or_boxes_used") is not True):
        raise ValueError("only authentic TRAIN and aggregate model-only predictions")
    source=train_report["true_train_displacement"]
    other=(prediction_report.get("predicted_signed_displacement") or
           prediction_report.get("predicted_displacement"))
    if (not other or source.get("count")!=512 or
            not 32<=other.get("count",0)<=1024):
        raise ValueError("sufficient matched authentic TRAIN and model outputs required")
    train_mode=source["most_common_vector_fraction"]
    model_mode=other["most_common_vector_fraction"]
    train_right=source["positive_x_fraction"]
    model_right=other["positive_x_fraction"]
    concentration=bool(model_mode >= .5 and train_mode <= .1)
    xdrift=bool(abs(model_right-train_right)>=.3)
    nearzero=bool(other["near_zero_fraction"]-source["near_zero_fraction"]>=.3)
    return {
        "schema":SCHEMA,
        "training_examples":source["count"],
        "observed_model_output_examples":other["count"],
        "train_mode_fraction":train_mode,
        "model_mode_fraction":model_mode,
        "train_positive_x_fraction":train_right,
        "model_positive_x_fraction":model_right,
        "repeat_single_displacement_warning":concentration,
        "rightward_direction_marginal_drift_warning":xdrift,
        "excess_near_zero_displacement_warning":nearzero,
        "any_direction_or_magnitude_collapse_warning":
            concentration or xdrift or nearzero,
        "posthoc_descriptive_warning_not_model_selection":True,
        "distribution_shift_is_not_causal_proof":True,
        "original_H61_success_required_for_GUI_qualification":True,
        "no_EVAL_task_targets_inspected":True,
    }

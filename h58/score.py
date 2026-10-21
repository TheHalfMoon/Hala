"""H58 score unchanged raw future model proposals on source-frozen PUBLIC DEV.

For original direction-INSENSITIVE sheet/highlight, reversed start/end may
satisfy the same task; direction-SENSITIVE resize must not get reverse credit.
This scorer runs no model and no browser. Real GUI success requires Edge replay.
"""
from __future__ import annotations
import json,math
from pathlib import Path
from h58.audit import audit,strict_file
SCHEMA="hala.h58.raw-public-dev-native-drag-attempts.v1"
ORIGINAL_SHA="a1c17c5dca028096bdd653de927fae18b1797a241ca3529646e10a1eb5f4f3de"

def parse_action(raw):
    if not isinstance(raw,str) or len(raw)>4096:return None
    def pairs(p):
        result={}
        for k,v in p:
            if k in result:raise ValueError("Duplicate model JSON key")
            result[k]=v
        return result
    try:
        j=json.loads(raw,object_pairs_hook=pairs,
                     parse_constant=lambda _:(_ for _ in ()).throw(ValueError("NaN/Infinity forbidden")))
    except (ValueError,RecursionError,TypeError,json.JSONDecodeError):
        return None
    if (not isinstance(j,dict) or set(j)!={"type","start_point","end_point",
         "coordinate_space","ordered"} or
        j["type"]!="drag" or j["coordinate_space"]!="source_image_pixels" or
        type(j["ordered"]) is not bool):
        return None
    for name in ("start_point","end_point"):
        v=j[name]
        if (not isinstance(v,list) or len(v)!=2 or
            type(v[0]) is not int or not 0<=v[0]<1024 or
            type(v[1]) is not int or not 0<=v[1]<640):
            return None
    return j

def score(root,proposals):
    s=audit(root)
    original=strict_file(Path(root)/"capture.json")
    if (not isinstance(proposals,dict) or set(proposals)!={"schema","source_sha256","origin","cases"} or
        proposals["schema"]!=SCHEMA or
        proposals["source_sha256"]!=s["original_source_sha256"] or
        proposals["origin"] not in ("synthetic_unit_fixture","provider_raw_model_outputs") or
        not isinstance(proposals["cases"],list) or len(proposals["cases"])!=24):
        raise ValueError("Exact original source-bound 24-case raw inference record required")
    expected={x["id"]:x for x in original["cases"]}
    found={}
    for c in proposals["cases"]:
        if not isinstance(c,dict) or set(c)!={"id","image_sha256","raw_output"}:
            raise ValueError("Exact raw case fields required")
        ident=c["id"]
        if (ident not in expected or ident in found or
            c["image_sha256"]!=expected[ident]["image_sha256"] or
            not isinstance(c["raw_output"],str) or len(c["raw_output"])>4096):
            raise ValueError("Source screenshot hash and unique scene ID required")
        found[ident]=c["raw_output"]
    if set(found)!=set(expected):raise ValueError("Missing or duplicated native drag scene")
    buckets={}
    for c in original["cases"]:
        label=c["domain"]+"/"+c["language"]
        b=buckets.setdefault(label,dict(n=0,valid=0,ordered_semantics_valid=0,
                                        geometrically_correct=0,reverse_accepted=0))
        b["n"]+=1
        pred=parse_action(found[c["id"]])
        if pred is None:continue
        b["valid"]+=1
        if pred["ordered"] is not c["ordered"]:continue
        b["ordered_semantics_valid"]+=1
        s0,e0=c["start_point"],c["end_point"]
        close=lambda a,v:math.dist(a,v)<22
        forward=close(pred["start_point"],s0) and close(pred["end_point"],e0)
        backward=close(pred["start_point"],e0) and close(pred["end_point"],s0)
        good=forward or (not c["ordered"] and backward)
        b["geometrically_correct"]+=int(good)
        b["reverse_accepted"]+=int(not c["ordered"] and backward)
    return dict(schema="hala.h58.public-dev-geometric-diagnostic.v1",
        source_sha256=ORIGINAL_SHA,origin=proposals["origin"],
        by_domain_language=buckets,scenes=24,
        total_valid=sum(b["valid"] for b in buckets.values()),
        ordered_semantics_valid=sum(b["ordered_semantics_valid"] for b in buckets.values()),
        geometrically_correct=sum(b["geometrically_correct"] for b in buckets.values()),
        reverse_accepted=sum(b["reverse_accepted"] for b in buckets.values()),
        original_public_dev_only=True,blind_test=False,
        actual_neural_inference_provenance_independently_verified=False,
        real_browser_model_task_success_verified=False,
        model_training_executed=False,benchmark_comparison_qualified=False)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="h58/assets")
    p.add_argument("--raw",required=True)
    args=p.parse_args()
    print(json.dumps(score(args.root,strict_file(args.raw,100000)),sort_keys=True))

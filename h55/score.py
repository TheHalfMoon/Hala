"""Score raw future Hala drag outputs on FIRST-PARTY PUBLIC DEV only.

No model is invoked. Accepts only unchanged per-scene raw strings and computes
geometry accuracy; real visual GUI completion requires separate Edge replay.
"""
from __future__ import annotations
import json,math,re
from pathlib import Path
from h55.audit import audit
SCHEMA="hala.h55.public-dev-raw-drag-attempts.v1"

def strict_json(raw):
    if not isinstance(raw,str) or len(raw)>8192:raise ValueError("Bounded raw response required")
    def pairs(x):
        result={}
        for k,v in x:
            if k in result:raise ValueError("Duplicate JSON object key")
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))

def parse_action(raw):
    try: obj=strict_json(raw)
    except (ValueError,TypeError,json.JSONDecodeError,RecursionError):return None
    if (not isinstance(obj,dict) or
        set(obj)!={"type","start_point","end_point","coordinate_space","ordered"} or
        obj["type"]!="drag" or obj["coordinate_space"]!="source_image_pixels" or
        obj["ordered"] is not True):return None
    for field in ("start_point","end_point"):
        p=obj[field]
        if (not isinstance(p,list) or len(p)!=2 or
            not all(type(v) is int and 0<=v<1024 for v in p) or p[1]>=640):
            return None
    return obj

def score(root,attempts):
    summary=audit(root)
    receipt=json.loads((Path(root)/"capture.json").read_text("utf8"))
    if (not isinstance(attempts,dict) or attempts.get("schema")!=SCHEMA or
        attempts.get("source_sha256")!=summary["original_source_sha256"] or
        attempts.get("origin") not in ("synthetic_unit_fixture","provider_raw_model_outputs") or
        not isinstance(attempts.get("cases"),list) or len(attempts["cases"])!=24):
        raise ValueError("Exact model/fixture source-bound 24 public DEV predictions required")
    images={c["id"]:c for c in receipt["cases"]}
    given={}
    for item in attempts["cases"]:
        if not isinstance(item,dict) or set(item)!={"id","image_sha256","raw_output"}:
            raise ValueError("Raw prediction must contain exactly immutable ID/image/raw output")
        k=item["id"]
        if (k in given or k not in images or item["image_sha256"]!=images[k]["image_sha256"]):
            raise ValueError("Unbound or repeated prediction/source screenshot")
        given[k]=item["raw_output"]
    if set(given)!=set(images):raise ValueError("Missing public DEV prediction")
    bylanguage={x:dict(n=0,valid=0,both_endpoints_hit=0,start_hit=0,end_hit=0) for x in ("ar","en")}
    for cid,source in images.items():
        r=bylanguage[source["language"]];r["n"]+=1
        action=parse_action(given[cid])
        if action is None:continue
        r["valid"]+=1
        start=action["start_point"];end=action["end_point"]
        hx,hy,hw,hh=source["handle_bbox"]
        start_hit=hx<=start[0]<=hx+hw and hy<=start[1]<=hy+hh
        # A true task success needs browser execution; pixel proximity is only proxy.
        end_hit=math.dist(end,source["end_point"])<=8
        r["start_hit"]+=int(start_hit)
        r["end_hit"]+=int(end_hit)
        r["both_endpoints_hit"]+=int(start_hit and end_hit)
    result=dict(schema="hala.h55.public-dev-geometry-diagnostic.v1",
       source_sha256=summary["original_source_sha256"],scenes=24,
       origin=attempts["origin"],by_language=bylanguage,
       total_valid=sum(i["valid"] for i in bylanguage.values()),
       both_endpoints_hit=sum(i["both_endpoints_hit"] for i in bylanguage.values()),
       public_dev_only=True,blind_test=False,
       actual_browser_model_task_success_verified=False,
       inference_provenance_independently_verified=False,
       official_benchmark_comparison_qualified=False,
       training_executed_by_scorer=False)
    return result
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="h55/assets")
    p.add_argument("--raw-attempts",required=True)
    args=p.parse_args()
    print(json.dumps(score(args.root,strict_json(Path(args.raw_attempts).read_text("utf8"))),
                    sort_keys=True))

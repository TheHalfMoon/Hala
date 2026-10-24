"""Owner-local read-only image SHA + source/frame + H61 oracle audit.

Requires a preexisting owner-private original DragOn EVAL cache, never retrieves
hidden TEST and never emits screenshot IDs/raw task JSON or image bytes.
"""
import json
from pathlib import Path
from h61.official_eval import DOMAINS,apply_frame,load_scenes
from h102.oracle import audit_eval_pair

SCHEMA="hala.h102.owner-original-eval-frame-geometry-audit.v1"

def run(source_root:Path,frame_root:Path,output:Path):
    source_root,frame_root,output=map(Path,(source_root,frame_root,output))
    if output.exists():
        raise FileExistsError("create-only aggregate owner-private evaluation audit")
    per_domain={}
    for domain in DOMAINS:
        folder=source_root/domain
        orig=load_scenes(folder/"scenes.json")
        if len(orig)!=20:
            raise ValueError("only original fixed indices 0..19 available for audit")
        framed=apply_frame(orig,frame_root/domain)
        if len(framed)!=20:
            raise ValueError("framed samples not paired")
        count=0
        for source,frame in zip(orig,framed):
            audit_eval_pair(source,frame)
            count+=1
        per_domain[domain]={"source_and_frame_sha_checked":count,
                            "unchanged_h61_oracle_successes":count,
                            "source_frame_transform_failures":0}
    result={"schema":SCHEMA,"status":"ORIGINAL_EVAL_FRAME_AND_ORACLE_PASS",
            "per_domain":per_domain,
            "original_source_scenes":sum(v["source_and_frame_sha_checked"] for v in per_domain.values()),
            "strict_oracle_scenes":sum(v["unchanged_h61_oracle_successes"] for v in per_domain.values()),
            "hidden_test_used":False,"model_training_executed":False,
            "model_performance_measured":False,"private_source_ids_exposed":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2),"utf8")
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--frames",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(Path(a.source),Path(a.frames),Path(a.output)),sort_keys=True))

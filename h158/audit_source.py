"""H158 independent original TodoMVC screenshot and source admission."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from h158.original_todomvc_source import PIN,SCHEMA,COUNTS,TOTAL

MANIFEST_SHA="d242938ceacfe5ff48cedbcbc71b2efaf99b0b36813f410a289b1f8f591a9e44"


def audit(owner):
    folder=Path(owner)/"private-h158"/"todomvc-source-once-v1"
    original=(folder/"manifest.json").read_bytes()
    if hashlib.sha256(original).hexdigest()!=MANIFEST_SHA:
        raise ValueError("H158 frozen source manifest changed")
    doc=json.loads(original)
    rows=doc.get("private_cases")
    if (doc.get("schema")!=SCHEMA or
        doc.get("status")!="H158_EXTERNAL_UNMODIFIED_TODOMVC_SOURCE_QUALIFIED" or
        doc.get("original_upstream_sha")!=PIN or
        doc.get("license")!="MIT" or
        doc.get("new_native_reference_tasks")!=TOTAL or
        doc.get("new_unique_original_images")!=TOTAL or
        doc.get("real_app_state_correct")!=TOTAL or
        doc.get("per_task_language_counts")!=COUNTS or
        doc.get("app_state_persists_after_reload") is not False or
        doc.get("model_inference_calls")!=0 or
        doc.get("optimizer_steps")!=0 or
        doc.get("h140_blind_calls")!=0 or
        not isinstance(rows,list) or len(rows)!=TOTAL):
        raise ValueError("H158 second app source not admitted")
    images=set()
    counts=Counter()
    for r in rows:
        if (r.get("task") not in COUNTS or
            r.get("language") not in COUNTS[r["task"]] or
            r.get("original_app_state_correct") is not True or
            r.get("state_persistent_after_reload") is not False or
            r.get("model_inference_calls")!=0 or
            not isinstance(r.get("instruction"),str) or
            not r["instruction"] or
            not isinstance(r.get("index"),int) or
            not isinstance(r.get("reference_point"),list) or
            len(r["reference_point"])!=2 or
            not all(type(x) in (int,float) and 0<=x<960 for x in
                    r["reference_point"][:1]) or
            not all(type(y) in (int,float) and 0<=y<720 for y in
                    r["reference_point"][1:])):
            raise ValueError("H158 original owner source case invalid")
        name=r["image"]
        if type(name) is not str or Path(name).name!=name or not name.endswith(".png"):
            raise ValueError("H158 private media path unsafe")
        data=(folder/"images"/name).read_bytes()
        if (len(data)<24 or data[:8]!=b"\x89PNG\r\n\x1a\n" or
            struct.unpack(">II",data[16:24])!=(960,720)):
            raise ValueError("H158 original native resolution differs")
        digest=hashlib.sha256(data).hexdigest()
        if digest in images or digest!=r["image_sha256"]:
            raise ValueError("H158 source media tampering or duplicates")
        images.add(digest)
        counts[(r["task"],r["language"])]+=1
    expected={(t,l):n for t,g in COUNTS.items() for l,n in g.items()}
    if len(images)!=TOTAL or counts!=expected:
        raise ValueError("H158 independent source strata not balanced")
    return {"status":"H158_SECOND_APP_SOURCE_AUDIT_PASS",
            "manifest_sha256":MANIFEST_SHA,
            "independent_source_app":"tastejs/todomvc",
            "upstream_sha":PIN,"license":"MIT",
            "new_images":len(images),
            "reference_success":TOTAL,
            "english_cases":8,"arabic_cases":8,
            "not_a_neural_model_score":True,
            "h140_blind_calls":0,
            "paid_compute":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    o=p.parse_args()
    print(json.dumps(audit(o.owner_root),sort_keys=True,indent=2))

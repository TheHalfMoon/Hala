"""Independently audit H172 frozen one-shot actual original browser DEV reward."""
import argparse,json,math
from collections import Counter
from pathlib import Path
from h172.source_dev import cases
from h172.infer_gpu import DEV_ZIP_SHA,VISION_WEIGHTS_SHA
from h172.native_replay import LOCK,OUTPUT_SCHEMA

def exact_mcnemar(wins,regressions):
    n=wins+regressions
    if n==0:return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(
        min(wins,regressions)+1))/2**n)

def audit(output,owner):
    doc=json.loads(Path(output).read_text("utf8"))
    lock=Path(owner)/"private-h172"/LOCK
    if (doc.get("schema")!=OUTPUT_SCHEMA
        or doc.get("status")!="H172_FRESH_ORIGINAL_DEV_NATIVE_ONE_SHOT_CONSUMED"
        or doc.get("source_dev_zip_sha256")!=DEV_ZIP_SHA
        or doc.get("trained_vision_sha256")!=VISION_WEIGHTS_SHA
        or doc.get("cases")!=32 or doc.get("real_gpu_proposals")!=64
        or doc.get("real_original_browser_native_reward") is not True
        or doc.get("owner_gold_on_gpu") is not False
        or doc.get("previous_consumed_ood_used") is not False
        or doc.get("new_dev_evaluated_once") is not True
        or doc.get("paid_gpu") is not False
        or doc.get("optimizer_steps")!=0
        or doc.get("flagship_35b_unchanged") is not True
        or not lock.is_file()):
        raise ValueError("H172 one-shot native original DEV evaluation evidence invalid")
    sealed=json.loads(lock.read_text("utf8"))
    if (sealed.get("status")!="H172_FRESH_DEV_PERMANENTLY_CONSUMED"
        or sealed.get("source_zip_sha256")!=DEV_ZIP_SHA):
        raise ValueError("H172 new DEV final consumption lock missing")
    outcomes=doc.get("private_outcomes")
    if not isinstance(outcomes,list) or len(outcomes)!=64:
        raise ValueError("H172 model-native 32 paired outcomes missing")
    frozen=cases()
    seen=set()
    counts=Counter()
    paired={}
    for item in outcomes:
        base=next((c for c in frozen if c["index"]==item.get("index")),None)
        role=item.get("role")
        key=(item.get("index"),role)
        if (base is None or role not in ("before","after") or key in seen
            or item.get("app")!=base["app"]
            or item.get("language")!=base["language"]
            or type(item.get("success")) is not bool
            or type(item.get("valid_click")) is not bool
            or (item["success"] and not item["valid_click"])):
            raise ValueError("H172 mislabeled, duplicated or invalid original browser outcome")
        seen.add(key)
        paired[key]=item["success"]
        counts[role+"/calls"]+=1
        counts[role+"/success"]+=int(item["success"])
        counts[role+"/"+base["app"]+"/success"]+=int(item["success"])
        counts[role+"/"+base["language"]+"/success"]+=int(item["success"])
    if (seen!={(c["index"],role) for c in frozen
               for role in ("before","after")}
        or counts["before/calls"]!=32 or counts["after/calls"]!=32
        or counts["before/success"]!=doc.get("before_success")
        or counts["after/success"]!=doc.get("after_success")
        or dict(counts)!=doc.get("counts")):
        raise ValueError("H172 true owner native success count provenance invalid")
    wins=sum(paired[(c["index"],"after")] and not paired[(c["index"],"before")]
             for c in frozen)
    regressions=sum(paired[(c["index"],"before")] and not paired[(c["index"],"after")]
                    for c in frozen)
    if doc.get("paired_wins")!=wins or doc.get("paired_regressions")!=regressions:
        raise ValueError("H172 paired win/regression claim did not recompute")
    return {
       "status":"H172_INDEPENDENT_ORIGINAL_NATIVE_32_PAIRED_DEV_VERIFIED",
       "before_success":counts["before/success"],
       "after_success":counts["after/success"],
       "paired_wins":wins,"paired_regressions":regressions,
       "two_sided_exact_mcnemar_p":exact_mcnemar(wins,regressions),
       "before_arabic":counts["before/ar/success"],
       "after_arabic":counts["after/ar/success"],
       "before_english":counts["before/en/success"],
       "after_english":counts["after/en/success"],
       "before_flatpickr":counts["before/flatpickr/success"],
       "after_flatpickr":counts["after/flatpickr/success"],
       "before_awesomplete":counts["before/awesomplete/success"],
       "after_awesomplete":counts["after/awesomplete/success"],
       "dev_cases":32,"dev_consumed":True,
       "unseen_original_app_ood_model_calls":0}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--owner",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.output,a.owner),indent=2,sort_keys=True))

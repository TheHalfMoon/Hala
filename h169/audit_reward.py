"""H169 independent one-shot DEV native-browser reward audit and exact paired test."""
import argparse,json,math
from collections import Counter
from pathlib import Path
from h169.source_dev import cases
from h169.infer_gpu import DEV_ZIP_SHA,VISION_WEIGHTS_SHA

def exact_mcnemar(wins,losses):
    n=wins+losses
    if n==0:
        return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(
        min(wins,losses)+1))/2**n)

def audit(output,owner):
    source=Path(output)
    doc=json.loads(source.read_text("utf8"))
    lock=Path(owner)/"private-h169"/"H169_NEW_ORIGINAL_DEV_ONE_SHOT_CONSUMED.lock"
    if (doc.get("schema")!="hala.h169.new-32-original-dev-owner-browser-native-reward.v1"
        or doc.get("status")!="H169_FRESH_ORIGINAL_DEV_NATIVE_PAIRED_ONE_SHOT_CONSUMED"
        or doc.get("source_dev_zip_sha256")!=DEV_ZIP_SHA
        or doc.get("trained_vision_sha256")!=VISION_WEIGHTS_SHA
        or doc.get("total")!=32 or doc.get("real_gpu_proposals")!=64
        or doc.get("evaluated_once") is not True
        or doc.get("old_consumed_ood_used") is not False
        or doc.get("model_weights_changed") is not False
        or doc.get("private_owner_gold_on_gpu") is not False
        or doc.get("reward_uses_native_state_not_gold_proximity") is not True
        or doc.get("paid_gpu") is not False
        or doc.get("flagship_35b_unchanged") is not True
        or not lock.is_file()):
        raise ValueError("H169 one-shot private native reward provenance failed")
    lk=json.loads(lock.read_text("utf8"))
    if (lk.get("source_sha")!=DEV_ZIP_SHA or
        lk.get("status")!="H169_NATIVE_DEV_CONSUMED_NO_REUSE"):
        raise ValueError("H169 new DEV replay consumed lock missing")
    rows=doc.get("private_outcomes")
    if not isinstance(rows,list) or len(rows)!=64:
        raise ValueError("H169 paired native task replay incomplete")
    valid=cases(); keys=set(); counts=Counter(); pair={}
    for result in rows:
        src=next((x for x in valid if x["index"]==result.get("index")),None)
        role=result.get("role")
        key=(result.get("index"),role)
        if (src is None or role not in ("before","after") or key in keys
            or result.get("app")!=src["app"]
            or result.get("language")!=src["language"]
            or type(result.get("success")) is not bool
            or type(result.get("valid_kind")) is not bool
            or result["success"] and not result["valid_kind"]):
            raise ValueError("H169 actual native app reward/role metadata malformed")
        keys.add(key);pair[key]=result["success"]
        counts[role+"/success"]+=int(result["success"])
        counts[role+"/calls"]+=1
        counts[role+"/"+src["language"]+"/success"]+=int(result["success"])
        counts[role+"/"+src["app"]+"/success"]+=int(result["success"])
    if (keys!={(r["index"],role) for r in valid
               for role in ("before","after")}
        or counts["before/calls"]!=32 or counts["after/calls"]!=32
        or counts["before/success"]!=doc.get("before_success")
        or counts["after/success"]!=doc.get("after_success")
        or dict(counts)!=doc.get("counts")):
        raise ValueError("H169 native reward counts or paired coverage altered")
    wins=sum(pair[(r["index"],"after")] and not pair[(r["index"],"before")]
             for r in valid)
    losses=sum(pair[(r["index"],"before")] and not pair[(r["index"],"after")]
               for r in valid)
    if doc.get("paired_wins")!=wins or doc.get("paired_regressions")!=losses:
        raise ValueError("H169 paired wins/regressions invalid")
    return dict(
        status="H169_INDEPENDENT_NEW_DEV_NATIVE_PAIRED_REWARD_VERIFIED",
        source="32 new same-family DEV, original app native state",
        before_success=counts["before/success"],after_success=counts["after/success"],
        paired_wins=wins,paired_regressions=losses,
        two_sided_exact_mcnemar_p=exact_mcnemar(wins,losses),
        before_arabic=counts["before/ar/success"],
        after_arabic=counts["after/ar/success"],
        before_english=counts["before/en/success"],
        after_english=counts["after/en/success"],
        before_type=counts["before/choices/success"],
        after_type=counts["after/choices/success"],
        before_drag=counts["before/nouislider/success"],
        after_drag=counts["after/nouislider/success"],
        dev_cases_per_role=32,dev_consumed=True,
        unseen_app_ood_evaluations=0)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--owner",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.output,a.owner),indent=2,sort_keys=True))

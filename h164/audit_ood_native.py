"""Independent H164 TWO truly new original-app actual neural reward auditor."""
import argparse,json,math
from pathlib import Path
from collections import Counter
from h164.replay_ood_native import SOURCE_SHA,OUTPUT,LOCK,PRED
from h164.new_ood_64 import PRIVATE,SCHEMA,cases
from h164.expand_288_train import digest
from h164.ood_gpu_infer import SOURCE_ZIP_SHA,PARENT_LORA_SHA

FIELDS={"image_sha","role","app","task","language",
        "valid","click_sent","real_native_original_app_success"}

def audit(owner,model_root):
    owner=Path(owner)
    content=(owner/PRIVATE/"manifest.json").read_bytes()
    source=json.loads(content)
    qa=json.loads((owner/"private-h164/H164_OOD_64_PIXEL_QA.json").read_text("utf8"))
    lock=json.loads((owner/"private-h164"/LOCK).read_text("utf8"))
    truth=json.loads((owner/"private-h164"/OUTPUT).read_text("utf8"))
    predicted=json.loads((Path(model_root)/PRED).read_text("utf8"))
    if (digest(content)!=SOURCE_SHA or
        source.get("schema")!=SCHEMA or
        qa.get("status")!="H164_OOD_64_NEW_APP_EXACT_BROWSER_PIXEL_REPLAY" or
        lock.get("status")!="H164_TWO_NEW_ORIGINAL_APPS_NATIVE_ONE_SHOT_CONSUMED" or
        lock.get("source_manifest_sha256")!=SOURCE_SHA or
        truth.get("status")!=
        "H164_COMPLETED_128_TRUE_NEW_APP_NATIVE_MODEL_EPISODES" or
        truth.get("original_browser_trials")!=128 or
        truth.get("original_app_cases")!=64 or
        truth.get("source_manifest_sha256")!=SOURCE_SHA or
        truth.get("owner_gold_sent_to_controller") is not False or
        truth.get("original_apps_not_in_train_or_dev") is not True or
        predicted.get("status")!="H164_COMPLETED_128_REAL_GPU_OOD_CALLS" or
        predicted.get("real_gpu_calls")!=128 or
        predicted.get("source_zip_sha256")!=SOURCE_ZIP_SHA or
        predicted.get("frozen_H161_adapter_sha256")!=PARENT_LORA_SHA or
        predicted.get("new_H164_adapter_sha256")!=truth.get("trained_adapter_sha256")
        or lock.get("trained_model_sha256")!=truth.get("trained_adapter_sha256")):
        raise ValueError("H164 native Pell/Pikaday app/model receipt provenance fails")
    expected=cases()
    frozen=source.get("private_cases")
    rows=truth.get("private_rows")
    if (len(frozen)!=64 or not isinstance(rows,list) or len(rows)!=128 or
        any(any(x.get(k)!=v for k,v in case.items())
            for case,x in zip(expected,frozen))):
        raise ValueError("H164 new real application role/case coverage invalid")
    pairs={(x["image_sha256"],role) for x in frozen
           for role in ("before","after")}
    observed={}
    for record in rows:
        pair=(record.get("image_sha"),record.get("role"))
        if (not isinstance(record,dict) or set(record)!=FIELDS or
            pair not in pairs or pair in observed or
            any(type(record[f]) is not bool for f in (
                "valid","click_sent","real_native_original_app_success")) or
            record["real_native_original_app_success"] and
            not (record["valid"] and record["click_sent"])):
            raise ValueError("H164 actual original app browser result invalid/tampered")
        observed[pair]=record
    if set(observed)!=pairs:
        raise ValueError("H164 owner-native model episodes not complete")
    total=Counter();group=Counter()
    for case in frozen:
        old=observed[(case["image_sha256"],"before")]
        new=observed[(case["image_sha256"],"after")]
        if any(row["app"]!=case["app"] or row["task"]!=case["task"]
               or row["language"]!=case["language"] for row in (old,new)):
            raise ValueError("H164 actual original Pell/Pikaday task/model mismatched")
        a=old["real_native_original_app_success"]
        b=new["real_native_original_app_success"]
        total["before"]+=int(a);total["after"]+=int(b)
        total["wins"]+=int(b and not a)
        total["regressions"]+=int(a and not b)
        group[case["app"]+"/before"]+=int(a)
        group[case["app"]+"/after"]+=int(b)
        group[case["language"]+"/before"]+=int(a)
        group[case["language"]+"/after"]+=int(b)
    dis=total["wins"]+total["regressions"]
    m=min(total["wins"],total["regressions"])
    p=min(1.,2*sum(math.comb(dis,k) for k in range(m+1))
          /(2**dis)) if dis else 1.
    if (truth.get("baseline_success")!=total["before"] or
        truth.get("trained_success")!=total["after"] or
        truth.get("paired_wins")!=total["wins"] or
        truth.get("paired_regressions")!=total["regressions"] or
        truth.get("exact_paired_sign_p")!=p or
        truth.get("per_app_language")!=dict(group)):
        raise ValueError("H164 measured genuine native new app reward unreproducible")
    return {
        "status":"H164_INDEPENDENT_TRUE_TWO_NEW_APP_NATIVE_REWARD_AUDITED",
        "baseline_real_ood_success":total["before"],
        "trained_real_ood_success":total["after"],
        "native_app_cases":64,"actual_browser_trials":128,
        "paired_wins":total["wins"],
        "paired_regressions":total["regressions"],
        "paired_sign_p":p,"per_app_language":dict(group),
        "original_app_independent_OOD":True,
        "h140_blind_calls":0,"paid_GPU":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--model-root",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.owner,a.model_root),indent=2,sort_keys=True))

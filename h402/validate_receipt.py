"""Validate H402 adversarial invented-GUI abstention research receipt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h402-private-synthetic-negative-eval"

def verify(root=ROOT):
    path=Path(root)
    binary=(path/"receipt.json").read_bytes()
    sha=hashlib.sha256(binary).hexdigest()
    if (path/"receipt.sha256").read_text("ascii").strip()!=sha:
        raise ValueError("H402 original Kaggle receipt SHA mismatch")
    def no_duplicates(pairs):
        data={}
        for key,val in pairs:
            if key in data: raise ValueError("H402 duplicate receipt field")
            data[key]=val
        return data
    doc=json.loads(binary,object_pairs_hook=no_duplicates)
    if (doc.get("schema")!="hala.h402.private-author-negative-native-v1" or
        doc.get("base_model")!="Qwen/Qwen3-VL-2B-Instruct" or
        doc.get("base_revision")!="89644892e4d85e24eaac8bacfd4f463576704203" or
        doc.get("adapter_sha256")!="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537" or
        doc.get("evaluator_is_independent") is not False or
        doc.get("real_app_gui") is not False or
        doc.get("source_rights_approved") is not False or
        doc.get("externally_sealed_ood") is not False or
        doc.get("training_performed_in_this_run") is not False or
        doc.get("model_weights_published") is not False):
        raise ValueError("H402 fabricated rights/OOD/model claims")
    status=doc.get("status")
    if status=="COMPLETE_AUTHOR_SYNTHETIC_STRESS_ONLY":
        if doc.get("n")!=24 or doc.get("device")!="Tesla T4":
            raise ValueError("H402 wrong invented stress test size/device")
        for mode in ("base","adapted"):
            results=doc.get(mode)
            if type(results) is not dict or set(results)!={"all","en","ar","positive","absent"}:
                raise ValueError("H402 missing scientific audit group")
            for role,n in (("all",24),("en",12),("ar",12),("positive",12),("absent",12)):
                group=results[role]
                if type(group) is not dict or set(group)!={"n","correct","invalid","false_clicks","abstentions"}:
                    raise ValueError("H402 malformed paired group")
                if any(type(group[k]) is not int for k in group):
                    raise ValueError("H402 count type invalid")
                if group["n"]!=n or any(not 0<=group[k]<=n for k in group if k!="n"):
                    raise ValueError("H402 count out of bounds")
            for key in ("correct","invalid","false_clicks","abstentions"):
                if (results["all"][key]!=results["positive"][key]+results["absent"][key] or
                    results["all"][key]!=results["en"][key]+results["ar"][key]):
                    raise ValueError("H402 bilingual/target absent totals mismatch")
            if results["positive"]["false_clicks"]!=0:
                raise ValueError("H402 false-click metric must be absent-target only")
    elif status=="FAILED_NO_GENUINE_OOD_QUALIFICATION":
        if type(doc.get("failure_class")) is not str:
            raise ValueError("H402 failed run lacks diagnostic class")
    else:
        raise ValueError("H402 unfinished run cannot qualify")
    return {"sha256":sha,"status":status,
            "evaluation_scope":"developer-invented only, NOT sealed external OOD",
            "base":doc.get("base"),"adapted":doc.get("adapted")}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))

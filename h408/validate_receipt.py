"""Verify H408 private first-party UI transfer research; NOT independent OOD."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h408-private-first-party-ui-composition"
ADAPTERS={
    "h403":"01f1c1846c76a5e1e785fcc431faed5ee2774638a0c46d84fedfbd2486ebcb4b",
    "h406":"26ac3b14380eb7dfb1709d67b771b7bfe3d383ec37d02e3cae5b1c48197936b6",
}

def verify(folder=ROOT):
    root=Path(folder)
    original=(root/"receipt.json").read_bytes()
    digest=hashlib.sha256(original).hexdigest()
    if digest!=(root/"receipt.sha256").read_text(encoding="ascii").strip():
        raise ValueError("H408 owner Kaggle receipt SHA mismatch")
    def reject_duplicate(items):
        result={}
        for k,v in items:
            if k in result:
                raise ValueError("H408 duplicate receipt JSON field")
            result[k]=v
        return result
    record=json.loads(original,object_pairs_hook=reject_duplicate)
    if (record.get("schema")!="hala.h408.private-first-party-ui-composition-transfer.v1"
        or record.get("official_base")!="Qwen/Qwen3-VL-2B-Instruct"
        or record.get("base_revision")!="89644892e4d85e24eaac8bacfd4f463576704203"
        or record.get("parent_adapters")!=ADAPTERS
        or record.get("source_family")!="first-party newly authored GUI-like button/cards/sidebar"
        or record.get("external_ood") is not False
        or record.get("third_party_gui_data") is not False
        or record.get("original_gui_train_rights_approved") is not False
        or record.get("new_training") is not False
        or record.get("model_weights_published") is not False
        or record.get("official_hala_model_qualified") is not False):
        raise ValueError("H408 cannot fabricate independent OOD, original screenshot rights or weights")
    status=record.get("status")
    if status=="COMPLETE_FIRST_PARTY_UI_COMPOSITION_DIAGNOSTIC":
        if (record.get("n")!=36 or record.get("device_name")!="Tesla T4"
            or type(record.get("original_image_prompt_fingerprint")) is not str
            or len(record["original_image_prompt_fingerprint"])!=64):
            raise ValueError("H408 original GUI source count or device invalid")
        for model in ("h403","h406"):
            groups=record.get(model)
            if type(groups) is not dict or set(groups)!={
                "all","en","ar","unique","absent","ambiguous"}:
                raise ValueError("H408 incomplete paired original GUI diagnostic")
            for name,n in (("all",36),("en",18),("ar",18),
                           ("unique",12),("absent",12),("ambiguous",12)):
                group=groups[name]
                if (type(group) is not dict
                    or set(group)!={"n","correct","invalid",
                                    "unsafe_clicks","abstentions","valid_points"}
                    or group.get("n")!=n):
                    raise ValueError("H408 original bilingual UI group invalid")
                if any(type(group.get(key)) is not int or not 0<=group[key]<=n
                    for key in ("correct","invalid","unsafe_clicks",
                                "abstentions","valid_points")):
                    raise ValueError("H408 impossible native coordinate/count")
                if (group["invalid"]+group["abstentions"]+group["valid_points"]!=n
                    or group["correct"]+group["invalid"]+group["unsafe_clicks"]>n):
                    raise ValueError("H408 invalid action partition")
            for key in ("correct","invalid","unsafe_clicks","abstentions",
                        "valid_points"):
                if (groups["all"][key]!=groups["en"][key]+groups["ar"][key]
                    or groups["all"][key]!=sum(groups[g][key] for g in
                        ("unique","absent","ambiguous"))):
                    raise ValueError("H408 native EN/AR/regime totals disagree")
            if (groups["unique"]["unsafe_clicks"]!=0
                or groups["unique"]["correct"]>groups["unique"]["valid_points"]
                or any(groups[k]["unsafe_clicks"]!=groups[k]["valid_points"]
                       or groups[k]["correct"]>groups[k]["abstentions"]
                       for k in ("absent","ambiguous"))):
                raise ValueError("H408 invalid missing/ambiguous refusal accounting")
    elif status=="FAILED_NO_MODEL_QUALIFICATION":
        if type(record.get("failure_class")) is not str:
            raise ValueError("H408 no failure reason")
    else:
        raise ValueError("H408 no valid actual evaluation record")
    return {"status":status,"sha256":digest,
            "source":"first-party generated GUI mock only, not external independent OOD",
            "h403":record.get("h403"),"h406":record.get("h406")}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))

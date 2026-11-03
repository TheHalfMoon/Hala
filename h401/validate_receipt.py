"""Verify H401 author-exposed native-click diagnostics, never Hala OOD."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h401-private-native-paired-research"

def verify(folder=ROOT):
    base=Path(folder)
    raw=(base/"receipt.json").read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=(base/"receipt.sha256").read_text("ascii").strip():
        raise ValueError("H401 Kaggle research receipt SHA mismatch")
    def duplicate_guard(pairs):
        doc={}
        for key,value in pairs:
            if key in doc: raise ValueError("H401 duplicate receipt JSON property")
            doc[key]=value
        return doc
    doc=json.loads(raw,object_pairs_hook=duplicate_guard)
    if (doc.get("schema")!="hala.h401.private-paired-native-invented-qwen3vl.v1" or
        doc.get("base_model")!="Qwen/Qwen3-VL-2B-Instruct" or
        doc.get("revision")!="89644892e4d85e24eaac8bacfd4f463576704203" or
        doc.get("adapter_sha256")!="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537" or
        doc.get("evaluation_is_independent") is not False or
        doc.get("official_hala_model_qualified") is not False or
        doc.get("source_rights_original_screenshots_approved") is not False or
        doc.get("all_images_are_first_party_generated") is not True or
        doc.get("external_ood") is not False or
        doc.get("model_weights_published") is not False):
        raise ValueError("H401 unqualified real GUI/OOD result claim")
    status=doc.get("status")
    if status=="COMPLETE_AUTHOR_SYNTHETIC_NATIVE_PAIRED_DIAGNOSTIC":
        if doc.get("sample_count")!=18 or doc.get("device")!="Tesla T4":
            raise ValueError("H401 native evaluation sample/device wrong")
        for mode in ("base","adapted"):
            result=doc.get(mode)
            if type(result) is not dict or set(result)!={"all","en","ar"}:
                raise ValueError("H401 incomplete paired measurement")
            for name,expected in (("all",18),("en",9),("ar",9)):
                scores=result[name]
                if (type(scores) is not dict or
                    set(scores)!={"n","valid_json","hits","unique_cells"} or
                    scores.get("n")!=expected or
                    any(type(scores.get(key)) is not int for key in
                        ("n","valid_json","hits","unique_cells")) or
                    not (0<=scores["hits"]<=scores["valid_json"]<=expected) or
                    not 0<=scores["unique_cells"]<=9):
                    raise ValueError("H401 invalid native action hit/count receipt")
            if (result["all"]["hits"]!=result["en"]["hits"]+result["ar"]["hits"] or
                result["all"]["valid_json"]!=result["en"]["valid_json"]+result["ar"]["valid_json"]):
                raise ValueError("H401 bilingual totals inconsistent")
    elif status=="FAILED_NO_OOD_QUALIFICATION":
        if type(doc.get("failure_class")) is not str:
            raise ValueError("H401 failed execution without source error class")
    else:
        raise ValueError("H401 incomplete private native evaluation")
    return {"status":status,"receipt_sha256":sha,
            "scope":"author-generated invented GUI ONLY, NOT external OOD",
            "base":doc.get("base"),"adapted":doc.get("adapted")}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))

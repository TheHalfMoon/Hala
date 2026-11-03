"""Verify a real Kaggle H398 receipt without GPU, network, or model files.

This is a narrowly scoped synthetic engineering receipt, not model qualification.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

EXPECTED_SCHEMA="hala.h398.private-invented-grid-gpu-smoke.v1"
RECEIPT=Path(__file__).resolve().parent/"verified_receipt"/"h398-synthetic-research"

def verify(root=RECEIPT):
    path=Path(root)
    original=(path/"receipt.json").read_bytes()
    expected=(path/"receipt.sha256").read_text(encoding="ascii").strip()
    actual=hashlib.sha256(original).hexdigest()
    if actual!=expected:
        raise ValueError("H398 Kaggle receipt SHA256 mismatch")
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError("H398 duplicate JSON receipt key")
            result[key]=value
        return result
    doc=json.loads(original,object_pairs_hook=unique_pairs)
    if (doc.get("schema")!=EXPECTED_SCHEMA or
        doc.get("status")!="SYNTHETIC_GPU_TRAINED_NOT_HALA_MODEL" or
        doc.get("optimizer_steps")!=160 or
        doc.get("train_n")!=960 or doc.get("dev_n")!=180 or
        doc.get("research_unseen_n")!=180 or
        doc.get("device_name")!="Tesla T4" or
        doc.get("languages")!=["en","ar"] or
        any(doc.get(flag) is not True for flag in (
            "no_external_gui_screenshots","no_h199_rights_bypass",
            "no_private_inputs","no_model_download",
            "no_ood_custodian_claim","no_flagship_weights"))):
        raise ValueError("H398 synthetic-only receipt scope drift")
    for name in ("baseline_dev","trained_dev","baseline_unseen","trained_unseen"):
        part=doc.get(name)
        if (not isinstance(part,dict) or part.get("n")!=180 or
            type(part.get("unique_grid_positions")) is not int or
            not 0<=part["unique_grid_positions"]<=9 or
            any(type(part.get(label)) not in (int,float) or
                not 0<=part[label]<=1 for label in ("all","en","ar"))):
            raise ValueError("H398 scored bilingual tiny-model metrics invalid")
    if (doc["trained_dev"]["all"]<doc["baseline_dev"]["all"] or
        doc["trained_unseen"]["all"]<doc["baseline_unseen"]["all"]):
        raise ValueError("H398 invented-grid experimental regression")
    return {"sha256":actual,"status":doc["status"],
            "scope":"synthetic toy only, NOT Hala model",
            "dev":doc["trained_dev"],"author_exposed_test":doc["trained_unseen"]}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))

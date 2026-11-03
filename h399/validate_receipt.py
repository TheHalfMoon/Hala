"""Validate private Kaggle H399 research outcome without opening adapter weights.

A signed-off Hala model or lawful screenshot dataset cannot be inferred from this.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h399-private-original-synthetic-2b-lora"

def verify(path=ROOT):
    path=Path(path)
    raw=(path/"receipt.json").read_bytes()
    claimed=(path/"receipt.sha256").read_text("ascii").strip()
    if hashlib.sha256(raw).hexdigest()!=claimed:
        raise ValueError("H399 private Kaggle receipt digest mismatch")
    def no_duplicates(pairs):
        out={}
        for name,value in pairs:
            if name in out:raise ValueError("H399 duplicated receipt property")
            out[name]=value
        return out
    record=json.loads(raw,object_pairs_hook=no_duplicates)
    if (record.get("schema")!="hala.h399.private-invented-gui-qwen3vl-2b-lora.v1" or
        record.get("base_model")!="Qwen/Qwen3-VL-2B-Instruct" or
        record.get("base_revision")!="89644892e4d85e24eaac8bacfd4f463576704203" or
        record.get("official_h199_train_manifest_admitted") is not False or
        record.get("official_hala_model_trained") is not False or
        record.get("independent_ood_gain_proven") is not False or
        record.get("private_kaggle_research_only") is not True or
        record.get("model_weights_published") is not False):
        raise ValueError("H399 untrusted receipt authorization or source scope")
    status=record.get("status")
    if status=="COMPLETED_REAL_2B_SYNTHETIC_RESEARCH_LORA":
        if (record.get("optimizer_steps")!=16 or
            record.get("train_n")!=24 or record.get("dev_n")!=6 or
            record.get("alternate_style_n")!=6 or
            record.get("device_name")!="Tesla T4" or
            type(record.get("language_only_trainable_parameters")) is not int or
            record["language_only_trainable_parameters"]<=0):
            raise ValueError("H399 optimizer or runtime incomplete")
        for metric in ("base_dev_loss","after_dev_loss",
                       "base_author_exposed_alt_loss",
                       "after_author_exposed_alt_loss"):
            value=record.get(metric)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError("H399 absent or invalid real model loss metric")
    elif status=="FAILED_NO_HALA_MODEL_QUALIFICATION":
        if type(record.get("failure_class")) is not str:
            raise ValueError("H399 failure receipt lacks reason type")
    else:
        raise ValueError("H399 unfinished Kaggle session is not evidence of training")
    return {"sha256":claimed,"status":status,"optimizer_steps":record.get("optimizer_steps"),
            "scope":"Qwen 2B original synthetic research, NOT official Hala release"}

if __name__=="__main__":
    print(json.dumps(verify(),sort_keys=True))

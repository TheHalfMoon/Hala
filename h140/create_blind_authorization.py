"""Owner-only creation of V3 DEV-admitted one-time BLIND inference certificate.

No raw DEV outputs or BLIND gold ever leave owner Windows. The resulting
small JSON contains only source/checkpoint SHA identifiers and a go/no-go.
"""
import argparse
import json
from pathlib import Path

from h140.evaluate_v3_gate import evaluate,actual_adapter_hash
from h140.blind_gpu_infer import (
    ADMISSION_SCHEMA,OWNER_BLIND_SHA,TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA,
)

def authorize(v2_dir,v3_dir,destination):
    destination=Path(destination)
    if (destination.exists() or
        not any(p.name=="private-h140" for p in destination.parents)):
        raise ValueError("H140 DEV authorization write-once and owner-private")
    gate=evaluate(v2_dir,v3_dir)
    if gate.get("status")!="ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND" or not all(
        gate.get("checks",{}).values()):
        raise ValueError("H140 failing DEV model cannot access one-time owner BLIND")
    v3=json.loads((Path(v3_dir)/"private-h140-v3-learning-evidence.json").read_text("utf8"))
    sha=actual_adapter_hash(v3_dir,v3)
    cert={
        "schema":ADMISSION_SCHEMA,
        "status":"ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND",
        "train_manifest_sha256":TRAIN_MANIFEST_SHA,
        "dev_manifest_sha256":DEV_MANIFEST_SHA,
        "owner_original_manifest_sha256":OWNER_BLIND_SHA,
        "adapter_sha256":sha,
        "private_only":True,
        "neural_blind_calls_before_authorization":0,
        "owner_dev_gate_all_conditions_pass":True,
    }
    with destination.open("x",encoding="utf8") as out:
        out.write(json.dumps(cert,sort_keys=True,indent=2)+"\n")
    return {
        "status":cert["status"],
        "adapter_sha256":sha,
        "owner_gold_ever_uploaded":False,
        "model_neural_blind_calls":0,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--v2",required=True)
    p.add_argument("--v3",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(authorize(a.v2,a.v3,a.output),sort_keys=True))

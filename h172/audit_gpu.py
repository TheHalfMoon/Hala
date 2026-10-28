"""Independent H172 private owner-free DEV model proposal audit.

Receipts are not real browser success. Never read owner reference gold here.
"""
import argparse,json,math
from collections import Counter
from pathlib import Path
from h172.infer_gpu import DEV_ZIP_SHA,VISION_WEIGHTS_SHA
from h167.combined_probe import LANGUAGE_WEIGHTS_SHA
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h164.native_action_contract import parse_action
from h172.source_dev import cases
from h166.vision_probe import MAX_PEAK_GIB,EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT

SOURCE_SHA="UNQUALIFIED"

def audit(root):
    root=Path(root)
    job=json.loads((root/"h172-paired-dev-job.json").read_text("utf8"))
    payload=json.loads((root/"hala-h172-private-paired-dev"/
        "h172-paired-dev-predictions.json").read_text("utf8"))
    if (job.get("schema")!="hala.h172.owner-private-paired-dev-kaggle-job.v1"
        or job.get("status")!="H172_PAIRED_32_NEW_DEV_REAL_GPU_PREDICTIONS_PASS"
        or job.get("source_commit")!=SOURCE_SHA
        or job.get("optimizer_steps")!=0 or job.get("real_gpu_calls")!=64
        or job.get("owner_gold_uploaded") is not False
        or job.get("old_ood_used") is not False
        or job.get("paid_gpu") is not False
        or job.get("weights_published") is not False):
        raise ValueError("H172 private source/job identity or safety failed")
    if (payload.get("schema")!="hala.h172.frozen-paired-original-dev-predictions.v1"
        or payload.get("status")!="H172_PRIVATE_64_REAL_GPU_DEV_PREDICTIONS_COMPLETE"
        or payload.get("source_dev_zip_sha256")!=DEV_ZIP_SHA
        or payload.get("frozen_language_sha256")!=LANGUAGE_WEIGHTS_SHA
        or payload.get("trained_vision_sha256")!=VISION_WEIGHTS_SHA
        or payload.get("base_model")!=MODEL_ID
        or payload.get("base_revision")!=MODEL_SHA
        or payload.get("paired_dev_cases")!=32
        or payload.get("real_gpu_calls")!=64
        or payload.get("optimizer_steps")!=0
        or payload.get("owner_gold_on_gpu") is not False
        or payload.get("previous_consumed_ood_used") is not False
        or payload.get("paid_gpu") is not False
        or payload.get("weights_published") is not False
        or payload.get("flagship_35b_unchanged") is not True
        or payload.get("versions")!={"torch":EXPECTED_TORCH,
            "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,
            "Pillow":"11.3.0"}
        or "T4" not in payload.get("gpu","")):
        raise ValueError("H172 frozen dual-adapter paired DEV GPU record failed")
    peak=payload.get("peak_allocated_gib")
    if (type(peak) not in (int,float) or not math.isfinite(peak)
        or not 0<peak<=MAX_PEAK_GIB
        or peak!=job.get("peak_allocated_gib")):
        raise ValueError("H172 T4 memory receipt mismatch")
    rows=payload.get("predictions")
    if not isinstance(rows,list) or len(rows)!=64:
        raise ValueError("H172 64 privately emitted model proposals required")
    frozen=cases()
    keys=set()
    counts=Counter()
    for row in rows:
        role=row.get("role")
        base=next((r for r in frozen
                   if r["index"]==row.get("case_index")),None)
        key=(row.get("case_index"),role)
        if (base is None or role not in ("before","after") or key in keys
            or any(row.get(k)!=base[k] for k in (
                "app","language","action_type"))
            or row.get("parsed")!=parse_action(row.get("raw"))):
            raise ValueError("H172 malformed or repeated model prediction")
        keys.add(key)
        counts[role+"/valid"]+=int(row["parsed"]["valid"])
        counts[role+"/calls"]+=1
    if (keys!={(x["index"],r) for x in frozen
                for r in ("before","after")}
        or counts["before/calls"]!=32 or counts["after/calls"]!=32):
        raise ValueError("H172 32 paired before/after input coverage missing")
    return dict(status="H172_INDEPENDENT_64_PAIRED_NEW_DEV_GPU_PROPOSALS_PASS",
        model_inferences=64,unique_dev_scenes=32,
        before_valid_outputs=counts["before/valid"],
        after_valid_outputs=counts["after/valid"],
        gpu_peak_allocated_gib=peak,optimizer_steps=0,
        native_browser_reward_measured=False,
        old_ood_used=False,paid_gpu=False)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.output),indent=2,sort_keys=True))

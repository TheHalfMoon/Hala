"""H157 independent owner-only original spatial action-source verification."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from h157.action_contract_v3 import SEQUENCES,parse_v3
from h157.collect_spatial_source import (
    APP_SHA,COUNTS,EXPECTED_TRAIN,EXPECTED_DEV,SCHEMA,prior_media_shas,
)
CHECKPOINT_SENTINEL="a"*64
EXPECTED_FIELD_SET={
    "split","task","seed","language","action_sequence",
    "image","image_sha256","instruction","training_supervision",
    "original_positive_native_reward","controller_uses_only_mouse_keyboard",
    "model_predictions_used",
}


def verify_png(raw):
    if (len(raw)<24 or raw[:8]!=b"\x89PNG\r\n\x1a\n" or
        raw[12:16]!=b"IHDR" or struct.unpack(">II",raw[16:24])!=(960,720)):
        raise ValueError("H157 original screenshot must be native 960x720 PNG")


def audit(owner):
    owner=Path(owner)
    folder=owner/"private-h157"/"source-spatial-train-dev-v1"
    raw=(folder/"manifest.json").read_bytes()
    doc=json.loads(raw)
    rows=doc.get("private_cases")
    total=EXPECTED_TRAIN+EXPECTED_DEV
    if (doc.get("schema")!=SCHEMA or
        doc.get("status")!="H157_FRESH_ORIGINAL_MULTI_TASK_SOURCE_READY" or
        doc.get("original_upstream_sha")!=APP_SHA or
        doc.get("original_license")!="MIT" or
        doc.get("train")!=EXPECTED_TRAIN or doc.get("dev")!=EXPECTED_DEV or
        doc.get("task_counts")!=COUNTS or
        doc.get("prior_original_sha_exclusions")!=146 or
        doc.get("new_unique_original_images")!=total or
        doc.get("positive_native_source_reference_episodes")!=total or
        doc.get("dev_is_blind") is not False or
        doc.get("neural_model_inferences")!=0 or
        doc.get("optimizer_steps")!=0 or
        doc.get("h140_sealed_blind_calls")!=0 or
        doc.get("no_paid_compute") is not True or
        doc.get("hala_flagship_35B_unchanged") is not True or
        not isinstance(rows,list) or len(rows)!=total):
        raise ValueError("H157 frozen original source manifest invalid")
    excluded=prior_media_shas(owner)
    media=set()
    seeds=set()
    strata=Counter()
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=EXPECTED_FIELD_SET or
            row["split"] not in COUNTS or
            row["task"] not in COUNTS[row["split"]] or
            row["language"]!="en" or
            row["action_sequence"]!=list(SEQUENCES[row["task"]]) or
            type(row["seed"]) is not int or row["seed"] in seeds or
            row["original_positive_native_reward"] is not True or
            row["controller_uses_only_mouse_keyboard"] is not True or
            row["model_predictions_used"] is not False or
            type(row["instruction"]) is not str or not row["instruction"]):
            raise ValueError("Original GUI source action/stratum contract violated")
        name=row["image"]
        if (not isinstance(name,str) or name!=Path(name).name or
            not name.endswith(".png")):
            raise ValueError("Unsafe H157 original screenshot pathname")
        picture=(folder/"images"/name).read_bytes()
        verify_png(picture)
        digest=hashlib.sha256(picture).hexdigest()
        if (digest!=row["image_sha256"] or
            digest in excluded or digest in media):
            raise ValueError("H157 source screenshot SHA duplicated/substituted")
        label=row["training_supervision"]
        try:
            actions=parse_v3(json.dumps(label),task=row["task"],
                             model_checkpoint_sha=CHECKPOINT_SENTINEL)
        except (TypeError,ValueError) as exc:
            raise ValueError("H157 original typed action supervision invalid") from exc
        if any(a.kind=="abstain" for a in actions):
            raise ValueError("H157 source reward cannot be supervision abstention")
        seeds.add(row["seed"])
        media.add(digest)
        strata[(row["split"],row["task"])]+=1
    if strata!={(role,task):quota for role,counts in COUNTS.items()
               for task,quota in counts.items()}:
        raise ValueError("H157 original source split/quota drift")
    return {
        "status":"H157_ORIGINAL_MULTI_LAYOUT_ACTION_SOURCE_AUDIT_PASS",
        "manifest_sha256":hashlib.sha256(raw).hexdigest(),
        "new_images":len(media),
        "exclusions":len(excluded),
        "train":EXPECTED_TRAIN,"dev":EXPECTED_DEV,
        "original_native_positive_reward_receipts":total,
        "action_task_families":len(SEQUENCES),
        "model_inference_calls":0,"optimizer_steps":0,
        "h140_blind_calls":0,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.owner_root),sort_keys=True,indent=2))

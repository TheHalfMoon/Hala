"""H113 owner-private source-verified original H71 TRAIN dual-point audit.

No model training, inference, model weights or EVAL/TEST media. The 512
original unique screenshots are exactly the already-qualified H105 plan.
This report only summarizes source labels and SHA; the private generated
SFT rows are retained on owner-private Kaggle for later qualification.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from h61.curriculum import read_shard,shard_roots,load_export as load_h71
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h105.curriculum import propose,COUNT as ORIGINAL_COUNT
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h113.dual_point import (derive_two_independent_points,QuantizationQuarantine)
from h61.official_eval import canonical

SCHEMA="hala.h113.private-original-h71-dual-point-target-quality-audit.v1"

def run(*,repo,scratch,output):
    repo,scratch,output=map(Path,(repo,scratch,output))
    if output.exists() or scratch.exists():
        raise ValueError("create-only isolated owner-private H113 audit")
    scratch.mkdir(parents=True,exist_ok=False)
    import os
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("H113 must use genuine owner-private free Kaggle CPU")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                       scratch/"source-H71",H71_CODE_SHA,rederive=True)
    if stage["status"]!="RESTORED_TRAIN_STAGE" or stage["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("real H71 original private TRAIN stage digest mismatch")
    root=scratch/"source-H71/export"
    path=root/"h61_drag_curriculum.jsonl"
    manifest=root/"h61_drag_curriculum_manifest.json"
    records=load_h71(path,json.loads(manifest.read_text("utf8")))
    original={}
    for shard in shard_roots(root):
        prefix=shard.relative_to(root).as_posix() if shard!=root else ""
        sources,_=read_shard(shard,prefix)
        for item in sources:
            if item["id"] in original:
                raise ValueError("duplicate original H71 TRAIN scene ID")
            original[item["id"]]=item
    if len(records)!=10000 or len(original)<10000:
        raise ValueError("expected exact original H71 verified 10k curriculum")
    plan=propose(records,original,ORIGINAL_COUNT)
    if (plan.sha256!=H105_QUALIFIED_PLAN_SHA or len(plan.indices)!=512
            or plan.summary.get("all_gates_passed") is not True):
        raise ValueError("cannot depart from owner-qualified original H105 TRAIN 512 plan")
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    excluded=Counter()
    langs=Counter()
    families=Counter()
    for i in plan.indices:
        item=records[i]
        src=original[item["id"]]
        try:
            two=derive_two_independent_points(item,src)
        except QuantizationQuarantine:
            excluded["relative_quantization_out_of_original_target_bbox"]+=1
            continue
        # Source identities/labels remain owner-private and never go into receipt.
        rows.extend(two)
        langs[item["language"]]+=1
        families[item["family"]]+=1
    eligible=len(rows)//2
    if len(rows)!=2*eligible or eligible+sum(excluded.values())!=512:
        raise ValueError("H113 original TRAIN/source pair conservation violated")
    data=output/"private-source-bound-dual-point-sft.jsonl"
    with data.open("wb") as stream:
        for row in rows:
            stream.write((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode("utf8"))
    sha=hashlib.sha256(data.read_bytes()).hexdigest()
    report={
        "schema":SCHEMA,"status":"ORIGINAL_H71_TRAIN_ONLY_DUAL_POINT_SOURCE_AUDITED",
        "source_train_digest":TRAIN_SHA,
        "original_h105_parent_512_plan_sha256":plan.sha256,
        "verified_original_source_screenshots_considered":512,
        "qualified_original_unique_images":eligible,
        "qualified_independent_point_completion_count":len(rows),
        "quarantine_counts":dict(sorted(excluded.items())),
        "selected_language_counts":dict(sorted(langs.items())),
        "selected_family_counts":dict(sorted(families.items())),
        "derived_private_point_sft_sha256":sha,
        "model_training_performed":False,
        "gpu_used":False,"model_weights_changed":False,
        "private_source_rows_not_in_receipt":True,
        "source_split":"TRAIN_ONLY",
        "no_eval_or_test_sources":True,
        "no_public_data_or_model_export":True,
        "commercial_use":False,
        "ready_to_train_model":False,
        "requires_independent_owner_receipt_replay":True,
    }
    (output/"h113-private-data-admission.json").write_bytes(canonical(report))
    return report

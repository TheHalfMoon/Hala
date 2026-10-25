"""Owner-private H123 original H71 CPU-only source and 1024 media admission."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import (ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,
                               locate_attached_train)
from h81.staged_resume import admit_stage
from h122.source_bound import admit_original_source_map,original_for_sft
from h123.schedule import derive,ORIGINAL,TRAIN,HOLDOUT
from h123.objective import parse
from training.artifacts import artifact_path,sha256_file

SCHEMA="hala.h123.private-original-1024-complex-TRAIN-CPU-source-only.v1"

def run(scratch:Path,output:Path):
    scratch,output=Path(scratch),Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("H123 only create-only PRIVATE CPU receipt")
    if not str(scratch).startswith("/kaggle/tmp/"):
        raise ValueError("H123 original TRAIN CPU scratch path required")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H123 original H71 TRAIN source SHA rejected")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    records=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    plan=derive(records,root)
    if len(plan.train)!=TRAIN or len(plan.heldout)!=HOLDOUT:
        raise ValueError("H123 exact 896 train +128 heldout source selection")
    originals=admit_original_source_map(root)
    seen=set()
    for cohort,rows,indices in (("train",plan.train,plan.train_indices),
                                ("heldout",plan.heldout,plan.heldout_indices)):
        for row,i in zip(rows,indices):
            source=original_for_sft(records[i],originals)
            if (row["original_source_id"]!=source["id"]
                    or row["image_sha256"]!=source["image_sha256"]
                    or row["image_path"]!=source["image_path"]
                    or source["split"]!="train" or source["id"] in seen
                    or row["family"]!=source["family"]):
                raise ValueError("H123 source TRAIN family/image/ID SHA drift")
            seen.add(source["id"])
            media=artifact_path(root,source["image_path"])
            if not media.is_file() or media.is_symlink() or sha256_file(media)!=source["image_sha256"]:
                raise ValueError("H123 original media real bytes SHA changed")
            dest=parse(row["completion"],source["image_width"],source["image_height"])
            box=source["end_bbox"]
            if dest is None or not (box[0]<=dest[0]<=box[2] and box[1]<=dest[1]<=box[3]):
                raise ValueError("H123 original source END quantization/box drift")
    if len(seen)!=ORIGINAL:
        raise ValueError("H123 expected original 1024 unique source media")
    result={**plan.summary,
            "status":"PASS_OWNER_H123_NEW_ORIGINAL_1024_COMPLEX_TRAIN_SOURCE",
            "original_h71_archive_SHA_verified":ARCHIVE_SHA,
            "original_h71_dataset_SHA_verified":TRAIN_SHA,
            "image_bytes_and_original_END_box_SHA_verified":ORIGINAL,
            "heldout_scores_seen_or_used":False,
            "actual_model_training_or_improvement_claimed":False,
            "paid_resources_used":False,"commercial_use":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
    return result

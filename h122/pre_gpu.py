"""H122 CPU source-first gate before allocating the free Kaggle GPU.

Prove ALL 512 frozen original H105 TRAIN screenshots and original END
boxes against H71 staged source bytes. This includes 64 disjoint
TRAIN-heldout images only for integrity admission, NOT optimizer use.
No EVAL/TEST data, no model load, no raw source coordinates in receipts.
"""
from collections import Counter
from pathlib import Path

from h119.training_plan import get_plan as full_original_plan
from h122.grid import verify_overlay
from h122.schedule import ORIGINAL,TRAIN,HOLDOUT,Schedule
from h122.source_bound import admit_original_source_map,original_for_sft
from training.artifacts import artifact_path,sha256_file

SCHEMA="hala.h122.cpu-pre-GPU-original-512-integrity.v1"


def qualify_before_gpu(rows,root,schedule,optimizer_plan):
    if (not isinstance(schedule,Schedule)
            or schedule.summary.get("source_split")!="TRAIN_ONLY"
            or schedule.summary.get("no_EVAL_or_TEST_source") is not True
            or schedule.summary.get("no_holdout_optimizer_input") is not True
            or len(schedule.train)!=TRAIN
            or len(schedule.heldout)!=HOLDOUT
            or len(schedule.train_original_indices)!=TRAIN
            or len(schedule.heldout_original_indices)!=HOLDOUT
            or len(set(schedule.train_original_indices))!=TRAIN
            or len(set(schedule.heldout_original_indices))!=HOLDOUT
            or set(schedule.train_original_indices)&set(schedule.heldout_original_indices)
            or (set(schedule.train_original_indices)|
                set(schedule.heldout_original_indices))!=set(range(ORIGINAL))
            or len(optimizer_plan.indices)!=TRAIN):
        raise ValueError("H122 frozen original TRAIN 448/64 image split missing")

    parent=full_original_plan(rows,root,ORIGINAL*2)
    if (optimizer_plan.summary.get("original_full512_train_sha256")!=parent.sha256
            or optimizer_plan.summary.get("h122_schedule_sha256")!=
               schedule.summary.get("schedule_sha256")
            or tuple(optimizer_plan.indices)!=tuple(
                parent.indices[i] for i in schedule.train_original_indices)):
        raise ValueError("H122 GPU optimizer selected nonoriginal or heldout image")
    sources=admit_original_source_map(root)
    image_sha_seen=set()
    source_ids_seen=set()
    counted={"train":Counter(),"heldout":Counter()}
    counted_languages={"train":Counter(),"heldout":Counter()}
    for cohort,positions,examples in (
        ("train",schedule.train_original_indices,schedule.train),
        ("heldout",schedule.heldout_original_indices,schedule.heldout)):
        for original_index,derived in zip(positions,examples):
            row=rows[parent.indices[original_index]]
            original=original_for_sft(row,sources)
            verify_overlay(original,derived)
            if (original["id"] in source_ids_seen or
                    original["image_sha256"] in image_sha_seen):
                raise ValueError("H122 original TRAIN or heldout source repeated")
            # Only read genuine staged H71 TRAIN media. Do not fetch
            # public EVAL or TEST and do not trust a filename alone.
            image=artifact_path(root,original["image_path"])
            if (not image.is_file() or image.is_symlink() or
                    sha256_file(image)!=original["image_sha256"]):
                raise ValueError("H122 staged original TRAIN screenshot bytes differ")
            source_ids_seen.add(original["id"])
            image_sha_seen.add(original["image_sha256"])
            counted[cohort][original["family"]]+=1
            counted_languages[cohort][original["language"]]+=1

    if (len(source_ids_seen)!=ORIGINAL or len(image_sha_seen)!=ORIGINAL
            or dict(counted["train"])!=schedule.summary["train_families"]
            or dict(counted["heldout"])!=schedule.summary["heldout_families"]
            or dict(counted_languages["train"])!=schedule.summary["train_languages"]
            or dict(counted_languages["heldout"])!=schedule.summary["heldout_languages"]):
        raise ValueError("H122 original 512 source family/language provenance drift")
    return {
        "schema":SCHEMA,
        "original_source_images_sha_verified":ORIGINAL,
        "optimizer_train_images_sha_verified":TRAIN,
        "excluded_heldout_train_images_sha_verified":HOLDOUT,
        "train_plan_sha256":optimizer_plan.sha256,
        "schedule_sha256":schedule.summary["schedule_sha256"],
        "original_parent_sha256":parent.sha256,
        "source_split":"TRAIN_ONLY",
        "no_EVAL_or_TEST_source_read":True,
        "no_heldout_optimizer_input":True,
        "model_initialized_or_trained":False,
        "raw_identifiers_or_target_boxes_exported":False,
    }

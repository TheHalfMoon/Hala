"""H122 exact original H71 TRAIN-only heldout neural input admission.

Reconstruct the 64 heldout H71 source rows from the same 512 H105
original TRAIN images, not from user-supplied EVAL boxes. Does not
inspect any official EVAL/TEST source. Never use these 64 for training.
"""
from __future__ import annotations

from pathlib import Path

from h113.dual_point import valid_box
from h119.training_plan import get_plan as original_full512_plan
from h122.grid import verify_overlay
from h122.source_bound import admit_original_source_map,original_for_sft
from h122.schedule import HOLDOUT,ORIGINAL,TRAIN,Schedule
from training.artifacts import sha256_file


def admit_original_train_heldout(rows, root, schedule):
    if (not isinstance(schedule,Schedule)
            or schedule.summary.get("source_split")!="TRAIN_ONLY"
            or schedule.summary.get("no_holdout_optimizer_input") is not True
            or schedule.summary.get("no_EVAL_or_TEST_source") is not True
            or len(schedule.train)!=TRAIN or len(schedule.heldout)!=HOLDOUT
            or len(schedule.train_original_indices)!=TRAIN
            or len(schedule.heldout_original_indices)!=HOLDOUT
            or set(schedule.train_original_indices)&set(schedule.heldout_original_indices)
            or set(schedule.train_original_indices)|set(schedule.heldout_original_indices)
               !=set(range(ORIGINAL))):
        raise ValueError("H122 exact frozen original 448/64 TRAIN-only split required")
    root=Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("H122 exact original H71 TRAIN media root required")
    original=original_full512_plan(rows,root,ORIGINAL*2)
    source_map=admit_original_source_map(root)
    chosen=[]
    seen_ids=set()
    seen_media=set()
    for original_index,derived in zip(
            schedule.heldout_original_indices,schedule.heldout):
        sft=rows[original.indices[original_index]]
        row=original_for_sft(sft,source_map)
        if (not verify_overlay(row,derived)
                or row.get("split")!="train"
                or row["id"] in seen_ids
                or row["image_sha256"] in seen_media
                or type(row.get("family")) is not str
                or type(row.get("language")) is not str):
            raise ValueError("H122 original source image or END geometry changed")
        image=(root/row["image_path"]).resolve()
        if (not image.is_relative_to(root) or not image.is_file()
                or image.is_symlink() or sha256_file(image)!=row["image_sha256"]):
            raise ValueError("H122 true original heldout TRAIN image bytes differ")
        from PIL import Image
        with Image.open(image) as picture:
            if (picture.size!=(row["image_width"],row["image_height"])
                    or picture.format not in ("PNG","JPEG")):
                raise ValueError("H122 real H71 source image dimensions/format differ")
        box=row["end_bbox"]
        if not valid_box(box,row["image_width"],row["image_height"]):
            raise ValueError("H122 only original source END boxes")
        case={
          "id":row["id"],"split":"train",
          "source_split":"TRAIN_ONLY_HELDOUT",
          "family":row["family"],"language":row["language"],
          "image_path":str(image),"image_sha256":row["image_sha256"],
          "image_width":row["image_width"],"image_height":row["image_height"],
          "prompt":derived["prompt"],
          "end_bbox":list(box)}
        chosen.append(case)
        seen_ids.add(row["id"])
        seen_media.add(row["image_sha256"])
    if (len(chosen)!=HOLDOUT or len(seen_media)!=HOLDOUT
            or len(set(c["family"] for c in chosen))!=5
            or set(c["language"] for c in chosen)!={"ar","en"}):
        raise ValueError("H122 exact unique 64 original TRAIN heldout media incomplete")
    return chosen

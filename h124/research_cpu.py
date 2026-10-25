"""H124 CPU-only original H71 fresh START+END source qualification.

Run private on owner Kaggle CPU. No GPU/model, no official EVAL/TEST,
no prior H113/H123 source reuse and no public source labels.
"""
import json
from pathlib import Path
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import (ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,
                               locate_attached_train)
from h81.staged_resume import admit_stage
from h122.source_bound import admit_original_source_map,original_for_sft
from h124.schedule import derive,ORIGINAL,TRAIN,HOLDOUT
from h124.objective import from_original,parse
from training.artifacts import artifact_path,sha256_file


SCHEMA="hala.h124.private-original-TRAIN1024-joint-geometry-CPU.v1"


def run(scratch,output):
    scratch,output=Path(scratch),Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("H124 private source admission is create-only")
    if not str(scratch).startswith("/kaggle/tmp/"):
        raise ValueError("H124 use owner-private Kaggle CPU scratch")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if (stage.get("status")!="RESTORED_TRAIN_STAGE" or
            stage.get("dataset_sha256")!=TRAIN_SHA):
        raise ValueError("H124 original H71 source SHA not admitted")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    records=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    schedule=derive(records,root)
    upstream=admit_original_source_map(root)
    ids=set();images=set()
    for cohort,roles,indices in (
            ("train",schedule.train,schedule.train_indices),
            ("heldout",schedule.heldout,schedule.heldout_indices)):
        for role,i in zip(roles,indices):
            source=original_for_sft(records[i],upstream)
            expected=from_original(records[i],source)
            if (source["id"] in ids or source["image_sha256"] in images
                    or source["split"]!="train"
                    or expected["completion"]!=role["completion"]
                    or expected["prompt"]!=role["prompt"]
                    or expected["image_sha256"]!=role["image_sha256"]
                    or expected["original_source_id"]!=role["original_source_id"]):
                raise ValueError("H124 new original TRAIN source image or joint START/END drift")
            media=artifact_path(root,source["image_path"])
            if (not media.is_file() or media.is_symlink()
                    or sha256_file(media)!=source["image_sha256"]):
                raise ValueError("H124 genuine original image SHA byte mismatch")
            pair=parse(role["completion"],source["image_width"],source["image_height"])
            if pair is None:
                raise ValueError("H124 original joint start-end-delta corruption")
            for k,name in enumerate(("start_bbox","end_bbox")):
                x,y=pair[k];b=source[name]
                if not (b[0]<=x<=b[2] and b[1]<=y<=b[3]):
                    raise ValueError("H124 original START/END source bbox drift")
            ids.add(source["id"]);images.add(source["image_sha256"])
    if len(ids)!=ORIGINAL or len(images)!=ORIGINAL:
        raise ValueError("H124 original unique source 1024 images not qualified")
    report={**schedule.summary,"schema":SCHEMA,
        "status":"PASS_OWNER_H124_NEW_ORIGINAL_TRAIN1024_JOINT_GEOMETRY_SOURCE",
        "original_H71_archive_sha256":ARCHIVE_SHA,
        "original_H71_dataset_sha256":TRAIN_SHA,
        "original_image_byte_and_two_box_SHA_verified":ORIGINAL,
        "new_train_images":TRAIN,"new_heldout_images":HOLDOUT,
        "model_optimizer_steps":0,"heldout_scored":False,
        "model_improvement_claimed":False,"paid_compute_used":False,
        "images_labels_or_weights_published":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n","utf8")
    return report

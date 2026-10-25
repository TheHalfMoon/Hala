"""H125 original never-used H71 TRAIN64 CPU source admission.

Only authorized owner free Kaggle CPU; no model downloads/training and no
original EVAL/TEST; source screenshots/true endpoint bboxes stay private.
"""
from pathlib import Path
import json
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h122.source_bound import admit_original_source_map,original_for_sft
from h113.dual_point import derive_two_independent_points
from h125.source import derive,TOTAL,OLD
from training.artifacts import artifact_path,sha256_file

SCHEMA="hala.h125.owner-private-original-unseen-TRAIN64-source-CPU.v1"

def run(scratch,output):
    scratch,output=Path(scratch),Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("H125 original owner CPU source proof create-only")
    if not str(scratch).startswith("/kaggle/tmp/"):
        raise ValueError("H125 private source requires original Kaggle CPU scratch")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H125 original TRAIN H71 archive cannot be qualified")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    cohort=derive(rows,root)
    originals=admit_original_source_map(root)
    seen=set();media=set()
    for i,case in zip(cohort.indices,cohort.cases):
        original=original_for_sft(rows[i],originals)
        pair=derive_two_independent_points(rows[i],original)
        if (case["id"] in seen or case["image_sha256"] in media
                or case["id"]!=original["id"]
                or case["image_sha256"]!=original["image_sha256"]
                or case["intent"]!=original["intent"]
                or pair[0]["original_source_id"]!=case["id"]
                or pair[1]["original_source_id"]!=case["id"]
                or original["split"]!="train"):
            raise ValueError("H125 new original source identity/START/END changed")
        file=artifact_path(root,original["image_path"])
        if not file.is_file() or file.is_symlink() or sha256_file(file)!=case["image_sha256"]:
            raise ValueError("H125 original screenshot byte SHA not equal source")
        seen.add(case["id"]);media.add(case["image_sha256"])
    if (len(seen)!=TOTAL or len(media)!=TOTAL
            or cohort.summary["prior_original_screenshots_excluded"]!=OLD):
        raise ValueError("H125 genuinely new untrained source originals unavailable")
    out={**cohort.summary,"schema":SCHEMA,
         "status":"PASS_OWNER_H125_NEW_64_ORIGINAL_TRAIN_GUI_SOURCE",
         "original_H71_dataset_SHA":TRAIN_SHA,
         "genuine_image_bytes_and_both_endpoints_verified":TOTAL,
         "optimizer_steps":0,"model_inference_calls":0,
         "prior_training_or_heldout_images_reused":False,
         "paid_compute_used":False,"source_or_weights_published":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n","utf8")
    return out

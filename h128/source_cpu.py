"""H128 private free CPU reference export for independent neural-score replay.

Original TRAIN reference boxes NEVER passed to GPU inference. This is a
separate source-derived ground-truth receipt held privately by the owner.
"""
import json
from pathlib import Path
from PIL import Image
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h128.source import propose,TOTAL,PREVIOUS
from training.artifacts import artifact_path,sha256_file

def run(scratch,output):
    scratch,output=Path(scratch),Path(output)
    if (output.exists() or not str(scratch).startswith("/kaggle/tmp/") or
        not str(output).startswith("/kaggle/working/")):
        raise ValueError("H128 private frozen source export create-only in owner Kaggle CPU")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                      scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H128 real original H71 TRAIN archive/source hash mismatch")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    schedule=propose(rows,root)
    if (len(schedule.cases)!=TOTAL or
        schedule.summary["all_previous_ID_and_SHAs_excluded"]!=PREVIOUS):
        raise ValueError("H128 frozen TRAIN16 source failed all previous source exclusions")
    private_cases=[]
    for case in schedule.cases:
        file=artifact_path(root,case["image_path"])
        if not file.is_file() or file.is_symlink() or sha256_file(file)!=case["image_sha256"]:
            raise ValueError("H128 original media SHA mismatch")
        with Image.open(file) as image:
            if image.size!=(case["image_width"],case["image_height"]) or image.format not in ("JPEG","PNG"):
                raise ValueError("H128 original image format/dimensions mismatch")
        private_cases.append({
            "id":case["id"],"image_sha256":case["image_sha256"],
            "family":case["family"],"image_width":case["image_width"],
            "image_height":case["image_height"],
            "start_bbox":case["start_bbox"],"end_bbox":case["end_bbox"]
        })
    output.parent.mkdir(parents=True,exist_ok=True)
    private={
        "schema":"hala.h128.owner-private-true-original-TRAIN16-source-replay.v1",
        "source_schedule_sha256":schedule.summary["schedule_sha256"],
        "source_cases_sha256":schedule.summary["cases_sha256"],
        "original_previous_excluded":PREVIOUS,
        "original_image_byte_sha_verified":16,
        "source_original_EVAL_TEST_used":False,
        "cases":private_cases,
    }
    output.write_text(json.dumps(private,indent=2,sort_keys=True)+"\n","utf8")
    return {"status":"PASS_H128_OWNER_PRIVATE_TRAIN16_SOURCE_REPLAY_CPU",
            "source_schedule_sha256":schedule.summary["schedule_sha256"],
            "source_cases_sha256":schedule.summary["cases_sha256"],
            "source_verified":16,"excluded_previous":PREVIOUS,
            "model_inference_calls":0,"optimizer_steps":0,"paid_compute_used":False}

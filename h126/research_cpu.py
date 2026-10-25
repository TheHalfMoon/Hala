"""H126 true original H71 unseen source64 admission and aggregate intent audit.

No model call, GPU, training, EVAL/TEST or release. All actual source
text, screenshots and original START/END boxes are strictly private.
"""
from pathlib import Path
import json
from PIL import Image
from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA,H71_CODE_SHA,TRAIN_SHA,locate_attached_train
from h81.staged_resume import admit_stage
from h122.source_bound import admit_original_source_map,original_for_sft
from h126.source_audit import propose,metadata_summary,TOTAL,PREVIOUS
from training.artifacts import artifact_path,sha256_file

def run(scratch,output):
    scratch,output=Path(scratch),Path(output)
    if output.exists() or not str(scratch).startswith("/kaggle/tmp/"):
        raise ValueError("H126 private CPU original screenshot audit create-only")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,
                     scratch/"H71",H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H126 genuine original H71 TRAIN source archive SHA mismatch")
    root=scratch/"H71/export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    selected=propose(rows,root)
    prior=admit_original_source_map(root)
    seen_ids=set();seen_media=set()
    for i,case in zip(selected.indices,selected.cases):
        src=original_for_sft(rows[i],prior)
        if (case["id"] in seen_ids or case["image_sha256"] in seen_media
            or src["id"]!=case["id"] or src["image_sha256"]!=case["image_sha256"]
            or src["intent"]!=case["intent"]
            or src["split"]!="train"):
            raise ValueError("H126 unseen original TRAIN source SHA/intent mismatch")
        media=artifact_path(root,src["image_path"])
        if not media.is_file() or media.is_symlink() or sha256_file(media)!=case["image_sha256"]:
            raise ValueError("H126 original new screenshot bytes differ")
        with Image.open(media) as image:
            if image.size!=(src["image_width"],src["image_height"]) or image.format not in ("JPEG","PNG"):
                raise ValueError("H126 original frame dimensions/media drift")
        for key in ("start_bbox","end_bbox"):
            if src[key]!=case[key]:
                raise ValueError("H126 source geometry labels differ from original")
        seen_ids.add(case["id"]);seen_media.add(case["image_sha256"])
    if len(seen_ids)!=TOTAL or len(seen_media)!=TOTAL or selected.summary["excluded_previous_source_screenshots"]!=PREVIOUS:
        raise ValueError("H126 true unique original TRAIN64 and all past exclusions failed")
    out=metadata_summary(selected)
    if out.get("semantic_identifiability_inferred") is not False:
        raise ValueError("H126 lexical source features cannot establish model visibility")
    out.update(status="PASS_H126_NEW_ORIGINAL_TRAIN64_SOURCE_LANGUAGE_GEOMETRY_CPU",
               actual_original_image_byte_sha_verified=TOTAL,
               original_H71_dataset_sha256=TRAIN_SHA,
               no_previous_2624_screenshots_used=True,
               new_optimizer_steps=0,new_neural_inference_calls=0,
               original_EVAL_or_TEST_used=False,
               paid_compute_used=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n","utf8")
    return out

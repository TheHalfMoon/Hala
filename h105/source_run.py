"""Owner-private H105 CPU: qualify a 512-sample *proposed* TRAIN curriculum.

No optimizer/model weights. Original H71 remains immutable. Full source
identity and actual media/oracle are verified for selected records.
"""
from __future__ import annotations
import json
from pathlib import Path
from h73.kaggle_h73_cpu import TRAIN_SHA,ARCHIVE_SHA,H71_CODE_SHA,locate_attached_train,require_cpu
from h81.staged_resume import admit_stage
from h61.curriculum import load_export,read_shard,shard_roots
from h99.sampler import build_balanced_plan
from h102.oracle import audit_train_pair
from h105.curriculum import propose,COUNT,STRATEGY

SCHEMA="hala.h105.owner-original-h71-512-geometry-curriculum-preflight.v1"

def run(scratch,output):
    require_cpu()
    scratch,output=Path(scratch),Path(output)
    if (scratch.exists() or output.exists() or
            not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")):
        raise ValueError("owner-private create-only CPU preflight required")
    admitted=admit_stage(locate_attached_train(),ARCHIVE_SHA,scratch,H71_CODE_SHA,rederive=True)
    if admitted.get("status")!="RESTORED_TRAIN_STAGE" or admitted["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("original H71 source archive identity/derived validation refused")
    source_root=scratch/"export"
    manifest=json.loads((source_root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(source_root/"h61_drag_curriculum.jsonl",manifest)
    if len(rows)!=10000 or manifest.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original immutable H71 10k required")
    source={}
    for shard in shard_roots(source_root):
        prefix=shard.relative_to(source_root).as_posix() if shard!=source_root else ""
        records,_=read_shard(shard,prefix)
        for original in records:
            if original["id"] in source:raise ValueError("duplicate original source record ID")
            source[original["id"]]=original
    if len(source)<10000:
        raise ValueError("source originals missing")
    proposal=propose(rows,source,COUNT)
    # H99 empirical source baseline: original H99 direction0 distribution
    # without any model inference; same frozen 10k and same audit source.
    h99=build_balanced_plan(rows,128)
    h99_major=max(
        sum(count for label,count in h99.summary["selected_geometry"].items()
            if label.startswith("dir"+str(n)+"_")) for n in range(8))
    h99_major_frac=h99_major/128
    checked_count=0
    selected_sha=set()
    for idx in proposal.indices:
        row=rows[idx]
        checked=audit_train_pair(row,source[row["id"]],source_root)
        selected_sha.add(checked["image_sha256"])
        checked_count+=1
    if checked_count!=COUNT or len(selected_sha)!=COUNT:
        raise ValueError("new proposed TRAIN media not independently original")
    selected=proposal.summary
    improved=(selected["largest_direction_fraction"]<h99_major_frac)
    gates=dict(selected["gates"])
    gates["h99_dominant_direction_share_reduced"]=improved
    report={
        "schema":SCHEMA,
        "status":("QUALIFIED_512_PROPOSED_TRAIN_GEOMETRY_COVERAGE_NOT_TRAINED"
                  if selected["all_gates_passed"] and improved
                  else "BLOCKED_H105_INSUFFICIENT_SOURCE_SPATIAL_COVERAGE"),
        "source_h71_sha256":TRAIN_SHA,
        "original_H71_rows":len(rows),
        "new_train_sample_count":COUNT,
        "original_source_screenshot_SHA_and_strict_oracle_verified":checked_count,
        "source_independent_images":len(selected_sha),
        "source_ids_exposed":False,
        "h99_baseline_dominant_direction_fraction":h99_major_frac,
        "proposed_dominant_direction_fraction":selected["largest_direction_fraction"],
        "proposed_language_counts":selected["language"],
        "proposed_family_counts":selected["family"],
        "proposed_direction_counts":selected["direction"],
        "proposed_distance_scale_counts":selected["scale"],
        "proposed_start_regions":selected["start_cells_occupied"],
        "proposed_end_regions":selected["end_cells_occupied"],
        "original_rows_quarantined_without_mutation":selected["original_source_quarantined_counts"],
        "proposal_sha256":proposal.sha256,
        "all_pre_registered_gates":gates,
        "full_original_10k_source_boundary_qualification_claimed":False,
        "model_training_executed":False,
        "model_weights_downloaded":False,
        "kaggle_gpu_used":False,
        "eval_or_hidden_test_used":False,
        "private_original_source_published":False,
        "model_better_than_baseline_claimed":False,
        "commercial_use":False,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"h105-private-original-train-curriculum-512.json").write_text(
        json.dumps(report,sort_keys=True,indent=2),"utf8")
    return report

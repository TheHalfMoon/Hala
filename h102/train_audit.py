"""Read-only Kaggle CPU admission and TRAIN geometry audit, original H71 only."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from h61.curriculum import read_shard,shard_roots,load_export
from h61.prompt_contract import source_body,drag_prompt
from h73.kaggle_h73_cpu import (TRAIN_SHA,H71_CODE_SHA,ARCHIVE_SHA,
                                locate_attached_train,require_cpu)
from h81.staged_resume import admit_stage
from h99.sampler import build_balanced_plan
from h102.oracle import audit_train_pair,in_bounds,inside,invalid_bbox_reasons

SCHEMA="hala.h102.owner-private-original-h71-train-geometry-integrity.v1"

def run(scratch:Path,output:Path):
    require_cpu()
    scratch,output=Path(scratch),Path(output)
    if (scratch.exists() or output.exists()
            or not str(scratch).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")):
        raise ValueError("create-only owner-private CPU scratch/output needed")
    admitted=admit_stage(locate_attached_train(),ARCHIVE_SHA,scratch,H71_CODE_SHA,rederive=True)
    if admitted.get("status")!="RESTORED_TRAIN_STAGE" or admitted["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("H71 original 10k source rederivation not admitted")
    original_root=scratch/"export"
    rows_path=original_root/"h61_drag_curriculum.jsonl"
    manifest=json.loads((original_root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(rows_path,manifest)
    if len(rows)!=10000 or manifest["dataset_sha256"]!=TRAIN_SHA:
        raise ValueError("the exact immutable original 10000 TRAIN is required")
    plan=build_balanced_plan(rows,128)
    chosen={rows[i]["id"]:rows[i] for i in plan.indices}
    if len(chosen)!=128:
        raise ValueError("H99 actual 128 source IDs not unique")
    source_by_id={}
    failures=Counter()
    invalid_bbox_details=Counter()
    invalid_bbox_families=Counter()
    invalid_bbox_selected_H99=0
    scanned=0
    ordered=Counter()
    true_distances=[]
    from math import hypot
    for shard in shard_roots(original_root):
        prefix=shard.relative_to(original_root).as_posix() if shard!=original_root else ""
        originals,_=read_shard(shard,prefix)
        for raw in originals:
            rid=raw["id"]
            if rid in source_by_id:
                # Retain ALL source rows only with unique ID; a repeated selected
                # source id is an immediate integrity refusal.
                raise ValueError("duplicate original TRAIN ID")
            source_by_id[rid]=raw
    if len(source_by_id)<10000:
        raise ValueError("missing original TRAIN source rows")
    for author in rows:
        source=source_by_id.get(author["id"])
        if source is None:
            failures["missing_source"]+=1
            continue
        scanned+=1
        w,h=source["image_width"],source["image_height"]
        if any(not in_bounds(source[k],w,h) for k in ("start_bbox","end_bbox")):
            failures["invalid_bbox"]+=1
            invalid_bbox_families[author["family"]]+=1
            invalid_bbox_selected_H99+=int(author["id"] in chosen)
            for k in ("start_bbox","end_bbox"):
                for reason in invalid_bbox_reasons(source[k],w,h):
                    invalid_bbox_details[k+":"+reason]+=1
            continue
        try:
            action=json.loads(author["completion"])
            good=(action==source["completion"] and
                  action["ordered"] is source["ordered"] and
                  source["image_path"]==author["image_path"] and
                  source["image_sha256"]==author["image_sha256"] and
                  author["prompt"]==drag_prompt(source_body(source["intent"],source["ordered"]),
                                                w,h,source["ordered"]))
            if not good:
                failures["label_or_prompt_divergence"]+=1
                continue
            if any(not inside(action[point],source[box]) for point,box in (
                    ("start_point","start_bbox"),("end_point","end_bbox"))):
                failures["target_outside_bbox"]+=1
            a,b=action["start_point"],action["end_point"]
            if a==b:
                failures["degenerate_reference"]+=1
            else:
                true_distances.append(round(hypot(b[0]-a[0],b[1]-a[1])))
            ordered["ordered" if source["ordered"] else "unordered"]+=1
        except (ValueError,KeyError,TypeError):
            failures["malformed_source"]+=1
    passes=0
    selected_languages,selected_families=Counter(),Counter()
    selected_shas=set()
    for row in chosen.values():
        source=source_by_id[row["id"]]
        proof=audit_train_pair(row,source,original_root)
        selected_shas.add(proof["image_sha256"])
        selected_languages[proof["language"]]+=1
        selected_families[proof["family"]]+=1
        passes+=1
    if len(selected_shas)!=128 or passes!=128:
        raise ValueError("H99 selected source screenshot/sample independence failed")
    # A failed source-label audit is a measured source-quality problem, not a
    # reason to silently filter samples or pretend Hala's GUI improved.
    result={"schema":SCHEMA,
            "status":("TRAIN_LABEL_GEOMETRY_GATE_PASS" if not failures
                      else "TRAIN_LABEL_GEOMETRY_GATE_FAIL"),
            "original_h71_train_sha256":TRAIN_SHA,
            "original_source_rows_scanned":scanned,
            "original_source_rows_available":len(source_by_id),
            "original_train_rows":len(rows),
            "original_10k_label_failures":dict(sorted(failures.items())),
            "invalid_bbox_reason_counts":dict(sorted(invalid_bbox_details.items())),
            "invalid_bbox_families":dict(sorted(invalid_bbox_families.items())),
            "invalid_bbox_present_in_H99_selected_prefix":bool(invalid_bbox_selected_H99),
            "ordered_counts":dict(sorted(ordered.items())),
            "unique_H99_screenshots_with_full_sha_media_and_oracle":len(selected_shas),
            "actual_H99_original_samples_checked":passes,
            "h99_sample_plan_sha256":plan.sha256,
            "selected_languages":dict(sorted(selected_languages.items())),
            "selected_families":dict(sorted(selected_families.items())),
            "nondegenerate_reference_count":len(true_distances),
            "source_images_exposed":False,
            "private_source_ids_exposed":False,
            "training_executed":False,
            "model_weights_downloaded":False,
            "evaluation_test_used":False,
            "commercial_use":False,
            "model_better_than_baseline_claimed":False}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h102-private-train-geometry-audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2),"utf8")
    return result

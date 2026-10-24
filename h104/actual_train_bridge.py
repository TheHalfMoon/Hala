"""H104 actual original H71 TRAIN -> pinned Qwen3-VL image processor oracle.

8 source-bound cases, owner-private free Kaggle CPU; only aggregate outputs.
Never edits frozen H71 labels, any H99 checkpoint, or the H61 scorer.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from h102.oracle import audit_train_pair
from h104.bridge import FrameBridge,roundtrip_oracle
from h73.kaggle_h73_cpu import TRAIN_SHA,ARCHIVE_SHA,H71_CODE_SHA,locate_attached_train,require_cpu
from h81.staged_resume import admit_stage
from h61.curriculum import load_export,shard_roots,read_shard
from h75.processor_audit import select_processor_audit_rows
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from training.scripts.baseline import MODEL_ID,MODEL_REVISION
from training.scripts.train_adapter import prepare_inputs

CASES=8
SCHEMA="hala.h104.owner-private-processor-frame-oracle.v1"

def run(scratch,output):
    import transformers
    from transformers import AutoProcessor
    require_cpu()
    scratch,output=Path(scratch),Path(output)
    if (scratch.exists() or output.exists() or
            not str(scratch).startswith("/kaggle/tmp/") or
            not str(output).startswith("/kaggle/working/")):
        raise ValueError("new bounded ephemeral private TRAIN CPU staging required")
    if transformers.__version__!="5.16.1":
        raise ValueError("pinned image processor version required")
    result=admit_stage(locate_attached_train(),ARCHIVE_SHA,scratch,H71_CODE_SHA,rederive=True)
    if result.get("status")!="RESTORED_TRAIN_STAGE" or result.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original H71 immutable source re-derive refused")
    root=scratch/"export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    if len(rows)!=10000 or manifest.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original private 10k H71 source and manifest needed")
    chosen=select_processor_audit_rows(rows,CASES)
    if len(chosen)!=8 or len({r["image_sha256"] for r in chosen})!=8:
        raise ValueError("eight source-distinct source-bound processor scenes required")
    required_ids={x["id"] for x in chosen}
    raw={}
    for shard in shard_roots(root):
        prefix=shard.relative_to(root).as_posix() if shard!=root else ""
        originals,_=read_shard(shard,prefix)
        for s in originals:
            if s["id"] in required_ids:
                if s["id"] in raw:raise ValueError("duplicate original TRAIN label identity")
                raw[s["id"]]=s
    if set(raw)!=required_ids:
        raise ValueError("original 8 H71 source label IDs missing")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    patch=getattr(processor.image_processor,"patch_size",None)
    if type(patch) is not int or patch<4 or patch>32:
        raise ValueError("actual pinned Qwen processor patch geometry missing")
    statuses=[]
    for row in chosen:
        source=raw[row["id"]]
        audit_train_pair(row,source,root)
        full,labels,weights=prepare_inputs(
            row,processor,root,manifest["token_budget"],
            native_drag_jpeg=True,supervision="coordinate_weighted",
            coordinate_weight=8.0,return_weights=True)
        grid=full["image_grid_thw"][0].tolist()
        if len(grid)!=3 or grid[0]!=1:
            raise ValueError("unexpected temporal/multi-image processor layout")
        sw,sh=source["image_width"],source["image_height"]
        pw,ph=int(grid[2])*patch,int(grid[1])*patch
        bridge=FrameBridge(sw,sh,pw,ph)
        scene={"id":source["id"],"domain":source["domain"],
               "image_width":sw,"image_height":sh,
               "task":{"start_bbox":source["start_bbox"],
                       "end_bbox":source["end_bbox"],
                       "ordered":source["ordered"]}}
        evidence=roundtrip_oracle(scene,bridge)
        statuses.append({"source_dims":[sw,sh],"processed_dims":[pw,ph],
                         "strict_source_success":evidence["valid_source_geometry"],
                         "roundtrip_error":evidence["max_pixel_roundtrip_error"],
                         "resized":not evidence["processor_frame_matches_source"]})
        del full,labels,weights
    failures=sum(not x["strict_source_success"] for x in statuses)
    report={"schema":SCHEMA,
            "status":("ORIGINAL_8_TRAIN_PROCESSOR_BRIDGE_STRICT_ORACLE_PASS"
                      if not failures else "ORIGINAL_PROCESSOR_BRIDGE_STRICT_ORACLE_FAIL"),
            "original_dataset_sha256":TRAIN_SHA,
            "source_rows":len(rows),"actual_processor_revision":MODEL_REVISION,
            "actual_processor_patch_size":patch,
            "original_H71_samples":len(statuses),
            "resized_cases":sum(x["resized"] for x in statuses),
            "strict_original_H61_roundtrip_success":len(statuses)-failures,
            "max_roundtrip_pixel_error":max(x["roundtrip_error"] for x in statuses),
            "frame_pairs":[{"source":x["source_dims"],"processor":x["processed_dims"]}
                           for x in statuses],
            "source_label_and_sha_oracle_verified":len(statuses),
            "source_mutation_performed":False,"Qwen_weight_files_downloaded":False,
            "hidden_test_or_eval_used":False,"training_executed":False,
            "model_success_improvement_claimed":False,
            "commercial_use":False,"private_ids_or_images_published":False}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h104-private-train-processor-roundtrip.json").write_text(
        json.dumps(report,sort_keys=True,indent=2),"utf8")
    return report

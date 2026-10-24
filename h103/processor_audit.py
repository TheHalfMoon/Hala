"""Read-only owner-private H71 -> actual pinned Qwen3-VL processor pipeline.

NO model weights, inference, optimizer, evaluation/test, paid GPU or publishing.
Demonstrate actual source-coordinate frame and completion-only numeric labels.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from h61.curriculum import load_export
from h73.kaggle_h73_cpu import TRAIN_SHA,ARCHIVE_SHA,H71_CODE_SHA,locate_attached_train,require_cpu
from h75.processor_audit import select_processor_audit_rows
from h81.staged_resume import admit_stage
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from training.scripts.baseline import MODEL_ID,MODEL_REVISION
from training.scripts.train_adapter import prepare_inputs
from h61.supervision import coordinate_token_flags

SCHEMA="hala.h103.owner-private-original-h71-processor-mask-integrity.v1"
CASES=8
PIXELS=1024*640


def grid_matches_source(width,height,patch_size,grid):
    """Image-grid of patch_size pixels must align exactly with source labels."""
    if (any(type(x) is not int or x<=0 for x in (width,height,patch_size))
            or not isinstance(grid,(tuple,list)) or len(grid)!=3
            or any(type(x) is not int or x<=0 for x in grid)):
        raise ValueError("positive integer source grid/patch dimensions required")
    t,gh,gw=grid
    return t==1 and gh*patch_size==height and gw*patch_size==width


def validate_supervision(full,prefix,labels,weights,coordinate_flags):
    """Check supervision boundary without needing torch or private text."""
    if (not all(isinstance(seq,list) for seq in (full,prefix,labels,weights,coordinate_flags))
            or not prefix or len(full)!=len(labels) or len(full)!=len(weights)):
        raise ValueError("bounded full/prefix/label/weight required")
    if full[:len(prefix)]!=prefix or len(full)<=len(prefix):
        raise ValueError("instruction/assistant token prefix mismatch")
    if labels[:len(prefix)]!=[-100]*len(prefix):
        raise ValueError("prompt/image-prefix leaked into labels")
    if any(v!=0 for v in weights[:len(prefix)]):
        raise ValueError("prompt/image-prefix leaked into coordinate loss")
    tail=full[len(prefix):]
    if labels[len(prefix):]!=tail:
        raise ValueError("supervised decoder targets differ from original completion")
    if len(coordinate_flags)!=len(tail) or not any(coordinate_flags):
        raise ValueError("no coordinate digit-target tokens in supervised completion")
    expected=[8.0 if is_coordinate else 1.0 for is_coordinate in coordinate_flags]
    if weights[len(prefix):]!=expected:
        raise ValueError("coordinate-weighted token loss mask drift")
    return {"supervised_tokens":len(tail),
            "coordinate_tokens":sum(coordinate_flags),
            "prompt_masked_tokens":len(prefix)}


def run(scratch:Path,output:Path):
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor
    require_cpu()
    if transformers.__version__!="5.16.1":
        raise ValueError("pinned Qwen processor transformers 5.16.1 required")
    scratch,output=Path(scratch),Path(output)
    if (scratch.exists() or output.exists() or
            not str(scratch).startswith("/kaggle/tmp/") or
            not str(output).startswith("/kaggle/working/")):
        raise ValueError("new owner-private ephemeral CPU staging and output only")
    stage=admit_stage(locate_attached_train(),ARCHIVE_SHA,scratch,H71_CODE_SHA,rederive=True)
    if stage.get("status")!="RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("H71 original source SHA/derived rows not admitted")
    root=scratch/"export"
    manifest=json.loads((root/"h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows=load_export(root/"h61_drag_curriculum.jsonl",manifest)
    if len(rows)!=10000 or manifest.get("dataset_sha256")!=TRAIN_SHA:
        raise ValueError("original immutable H71 10k required")
    samples=select_processor_audit_rows(rows,CASES)
    if len(samples)!=CASES or len({r["image_sha256"] for r in samples})!=CASES:
        raise ValueError("representative independent bilingual/family scenes required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    patch=getattr(processor.image_processor,"patch_size",None)
    if type(patch) is not int or not 4<=patch<=32:
        raise ValueError("pinned image processor patch geometry unqualified")
    statuses=[]
    for row in samples:
        # prepare_inputs loads exactly one original SHA-bound screenshot, with
        # the same shared prompt contract and loss masks as real H99 trainer.
        full,labels,weights=prepare_inputs(
            row,processor,root,manifest["token_budget"],
            native_drag_jpeg=True,supervision="coordinate_weighted",
            coordinate_weight=8.0,return_weights=True)
        source=root/row["image_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row["image_sha256"]:
            raise ValueError("training image mutated after processor preparation")
        with Image.open(source) as im:
            width,height=im.size
            image=im.convert("RGB")
        if width*height>2_000_000:
            raise ValueError("unbounded original source screenshot")
        messages=[{"role":"user","content":[
            {"type":"image"},{"type":"text","text":row["prompt"]}]}]
        prefix_text=processor.apply_chat_template(messages,tokenize=False,
                                                 add_generation_prompt=True)
        prefix=processor(text=[prefix_text],images=[image],return_tensors="pt")
        if "image_grid_thw" not in full or "image_grid_thw" not in prefix:
            raise ValueError("missing actual Qwen image patch grid")
        fgrid=full["image_grid_thw"][0].tolist()
        pgrid=prefix["image_grid_thw"][0].tolist()
        if fgrid!=pgrid or not grid_matches_source(width,height,patch,[int(v) for v in pgrid]):
            raise ValueError("processor silently resizes the pixel frame relative to source labels")
        if not ("pixel_values" in full and "pixel_values" in prefix
                and torch.equal(full["pixel_values"],prefix["pixel_values"])):
            raise ValueError("train/eval prefix image tensor mismatch")
        full_ids=full["input_ids"][0].tolist()
        prefix_ids=prefix["input_ids"][0].tolist()
        label_ids=labels[0].tolist()
        weights_list=weights[0].tolist()
        flags=coordinate_token_flags(full_ids[len(prefix_ids):],
                                     processor.tokenizer.decode)
        evidence=validate_supervision(full_ids,prefix_ids,label_ids,weights_list,flags)
        if (len(full_ids)>manifest["token_budget"]
                or evidence["coordinate_tokens"]<4
                or evidence["supervised_tokens"]<evidence["coordinate_tokens"]):
            raise ValueError("original H71 coordinate supervision token budget invalid")
        statuses.append({"source_pixels":width*height,
                         "processor_grid":fgrid,
                         "source_frame_match":True,
                         "train_eval_pixel_tensor_identity":True,
                         **evidence,
                         "prompt_masked":True,"coordinate_weight":8.0})
        del full,labels,weights,prefix,image
    result={"schema":SCHEMA,
            "status":"PINNED_H71_QWEN3VL_PROCESSOR_MASK_GATE_PASS",
            "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
            "transformers_version":transformers.__version__,
            "source_h71_sha256":TRAIN_SHA,
            "original_train_records":len(rows),
            "audited_original_bilingual_family_screenshots":len(samples),
            "source_frames_preserved":sum(x["source_frame_match"] for x in statuses),
            "train_eval_pixel_tensors_identical":len(statuses),
            "instruction_prefix_masked_cases":len(statuses),
            "coordinate_weight_verified_cases":len(statuses),
            "min_coordinate_supervision_tokens":min(x["coordinate_tokens"] for x in statuses),
            "min_source_pixels":min(x["source_pixels"] for x in statuses),
            "max_source_pixels":max(x["source_pixels"] for x in statuses),
            "patch_size_pixels":patch,
            "sample_language_counts":{k:sum(r["language"]==k for r in samples) for k in ("ar","en")},
            "sample_family_count":len({r["family"] for r in samples}),
            "H71_full_10k_boundary_exception_remains_unqualified":True,
            "GPU_used":False,"model_weights_downloaded":False,
            "training_executed":False,"model_inference_performed":False,
            "model_quality_improved_claimed":False,"evaluation_or_hidden_test_used":False,
            "raw_source_records_exposed":False,
            "commercial_use":False}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h103-private-processor-mask-audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2),"utf8")
    return result

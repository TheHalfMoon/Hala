"""H57 source-verified TRUE mixed ordered/unordered DragOn LoRA records.

Original H Company JPEG and JSON stay PRIVATE Kaggle. GitHub holds only source,
per-file SHA receipts and synthetic-only tests. This module never runs a model.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from h57.import_train_multidomain import (REV,SCHEMA as SOURCE_SCHEMA,
    audit as audit_original,sha,canonical,DOMAINS)
from training.artifacts import sha256_file,strict_json

SCHEMA="hala.h57.native-multidomain-drag-sft.v1"
ORIGINAL_TRAIN_SHA="6f7e59cb86d1c7ac80fe70ade8447e11990b420586e505631fb7a4a6a69c38bf"
ORIGINAL_SCREENS=117
ORIGINAL_ROWS=1586
ORDERED_TRUE=932
ORDERED_FALSE=654
PRIVATE_KAGGLE_DATASET="abdulazizshehri/hala-dragon-multidomain-train-private"
FAMILY_PREFIX="hcompany_dragon_"
RESERVED_DEV_FAMILY="original_disjoint_mixed_drag_holdout_reserved"
SOURCE_RECEIPT=Path(__file__).resolve().parents[1]/"h57/source_evidence.json"
ROWS_FILE="h57_native_drag_sft.jsonl"
MANIFEST_FILE="h57_native_drag_training_manifest.json"

def verify_source(root,receipt_path=None):
    root=Path(root)
    receipt_path=Path(SOURCE_RECEIPT if receipt_path is None else receipt_path)
    source=audit_original(root)
    external=strict_json(receipt_path.read_text("utf8"))
    manifest=strict_json((root/"manifest.json").read_text("utf8"))
    if (source["original_train_sha256"]!=ORIGINAL_TRAIN_SHA or
        source["screens"]!=ORIGINAL_SCREENS or source["actions"]!=ORIGINAL_ROWS or
        source["ordered_true"]!=ORDERED_TRUE or source["ordered_false"]!=ORDERED_FALSE or
        manifest.get("original_revision")!=REV or
        external.get("pinned_source_revision")!=REV or
        external.get("original_train_jsonl_sha256")!=ORIGINAL_TRAIN_SHA or
        external.get("original_source_manifest_sha256")!=sha((root/"manifest.json").read_bytes()) or
        external.get("kaggle_private_dataset")!=PRIVATE_KAGGLE_DATASET or
        external.get("kaggle_private_provider_confirmed") is not True or
        external.get("private_train_original_labels_remote_verified") is not True or
        external.get("third_party_raw_images_committed_to_public_github") is not False or
        manifest.get("only_original_train") is not True or
        manifest.get("benchmark_eval_or_test_used") is not False):
        raise ValueError("H57 requires independently verified immutable PRIVATE TRAIN identity")
    reference=external.get("source_image_task_sha256")
    original=manifest.get("original_paired_image_task_members")
    if (not isinstance(reference,list) or not isinstance(original,list) or
        len(reference)!=len(original) or len(original)!=ORIGINAL_SCREENS):
        raise ValueError("Incomplete signed per-original-JPEG/task digest catalog")
    wanted_keys=set(reference[0]) if reference else set()
    keys={"domain","image_id","image_sha256","task_sha256",
          "original_image_source_offset","original_task_source_offset","task_count"}
    if wanted_keys!=keys:raise ValueError("Unexpected signed per-source receipt schema")
    for x,e in zip(reference,original):
        if (set(x)!=keys or
            x!={k:e[k] for k in keys}):
            raise ValueError("Source JPEG/task/original byte offset SHA differs from signed catalog")
    return manifest,sha256_file(receipt_path)

def original_to_sft(root,receipt_path=None):
    root=Path(root)
    manifest,receipt_sha=verify_source(root,receipt_path)
    raw=(root/"train_drag_multidomain.jsonl").read_bytes()
    original=[strict_json(line) for line in raw.decode("utf8").splitlines()]
    if len(original)!=ORIGINAL_ROWS:raise ValueError("Missing original external TRAIN action")
    rows=[]
    for src in original:
        domain=src.get("domain")
        if (domain not in DOMAINS or src.get("split")!="train" or
            src.get("action_type")!="drag" or type(src.get("ordered")) is not bool or
            src.get("language")!="en" or
            type(src.get("image_width")) is not int or type(src.get("image_height")) is not int):
            raise ValueError("Only original H57 native mixed-domain TRAIN allowed")
        completion=src.get("completion")
        if (not isinstance(completion,dict) or
            set(completion)!={"type","start_point","end_point","coordinate_space","ordered"} or
            completion["type"]!="drag" or
            completion["coordinate_space"]!="source_image_pixels" or
            completion["ordered"] is not src["ordered"]):
            raise ValueError("Source ordered true/false cannot be overwritten in model labels")
        order="direction-sensitive" if src["ordered"] else "direction-insensitive"
        prompt=(f"Inspect the ORIGINAL {src['image_width']}x{src['image_height']} "
                f"GUI screenshot. Complete the {domain} selection/resize action: {src['intent']}\n"
                "Return ONLY one JSON object: type=drag, start_point=[x,y], "
                "end_point=[x,y], coordinate_space=source_image_pixels, "
                f"ordered={str(src['ordered']).lower()}. The task is {order}. "
                "Use integer coordinates from the ORIGINAL screenshot, not the resized "
                "vision tokens. Never click, browse or access outside services.")
        rows.append(dict(id=src["id"],split="train",
            family=FAMILY_PREFIX+domain+"_train",language="en",
            prompt=prompt,completion=json.dumps(completion,ensure_ascii=False,
                                 sort_keys=True,separators=(",",":")),
            image_path=src["image_path"],image_sha256=src["image_sha256"]))
    if len({r["id"] for r in rows})!=ORIGINAL_ROWS:
        raise ValueError("Duplicated original source training identity")
    ordered=[strict_json(row["completion"])["ordered"] for row in rows]
    if sum(ordered)!=ORDERED_TRUE or sum(not q for q in ordered)!=ORDERED_FALSE:
        raise ValueError("Original source order-sensitive balance changed")
    return rows,manifest,receipt_sha

def derived_artifacts(root,receipt_path=None):
    rows,source,receipt_sha=original_to_sft(root,receipt_path)
    blob=b"".join(json.dumps(r,ensure_ascii=False,sort_keys=True,
               separators=(",",":"),allow_nan=False).encode("utf8")+b"\n" for r in rows)
    media=sorted([dict(path=f"images/{e['domain']}/{e['image_id']}.jpg",
                  family=FAMILY_PREFIX+e["domain"]+"_train",
                  sha256=e["image_sha256"])
                  for e in source["original_paired_image_task_members"]],key=lambda x:x["path"])
    manifest=dict(schema="hala.training.manifest.v1",source_schema=SCHEMA,
        split="train",rights_cleared=True,source_kind="rights_cleared_public",
        source_dataset="Hcompany/DragOn",source_revision=REV,
        source_split="train/{sheet,slide_resize,text_highlight}.tar",
        source_license="Apache-2.0",raw_original_jsonl_sha256=ORIGINAL_TRAIN_SHA,
        original_source_receipt_sha256=receipt_sha,
        dataset_sha256=sha(blob),token_budget=4096,
        families=[FAMILY_PREFIX+k+"_train" for k in sorted(DOMAINS)],
        held_out_families=[RESERVED_DEV_FAMILY],
        independent_mixed_drag_dev_available=False,
        media=media,unique_source_images=ORIGINAL_SCREENS,train_rows=ORIGINAL_ROWS,
        ordered_true=ORDERED_TRUE,ordered_false=ORDERED_FALSE,
        native_action_type="drag",coordinate_space="source_image_pixels",
        original_source_domain_counts={name:{k:stats[k] for k in
            ("accepted_source_images","native_drag_examples","ordered_true","ordered_false")}
           for name,stats in source["domains"].items()},
        no_benchmark_eval_source=True,training_executed=False,
        model_improvement_qualified=False)
    return blob,manifest,rows

def load_export(examples_path,manifest):
    examples_path=Path(examples_path)
    if examples_path.name!=ROWS_FILE or examples_path.is_symlink() or not isinstance(manifest,dict):
        raise ValueError("H57 source-bound private mixed native drag SFT only")
    blob,expected,rows=derived_artifacts(examples_path.parent)
    if manifest!=expected or examples_path.read_bytes()!=blob:
        raise ValueError("H57 source train labels/media/derived SFT were modified")
    return rows

def export(root):
    root=Path(root)
    a,b=root/ROWS_FILE,root/MANIFEST_FILE
    if a.exists() or b.exists():raise FileExistsError("H57 model candidate must be create-only")
    blob,manifest,rows=derived_artifacts(root)
    with a.open("xb") as f:f.write(blob)
    try:
        with b.open("xb") as f:f.write(canonical(manifest))
    except BaseException:
        a.unlink()
        raise
    if load_export(a,manifest)!=rows:
        raise ValueError("H57 create-only original source-to-Qwen conversion failed")
    return dict(schema=SCHEMA,verified_original_screens=ORIGINAL_SCREENS,
        genuine_native_drag_rows=len(rows),ordered_true=ORDERED_TRUE,
        ordered_false=ORDERED_FALSE,derived_train_sha256=manifest["dataset_sha256"],
        model_training_executed=False,independent_model_evaluation_completed=False)

def validate_mixed_drag_baseline(baseline):
    # An honest receipt schema preflight, NOT third-party independent attestation.
    if (not isinstance(baseline,dict) or
        baseline.get("evidence")!="real_inference" or
        baseline.get("task_kind")!="mixed_ordered_unordered_drag" or
        baseline.get("held_out_source")!="original_source_disjoint_multidomain_drag_not_training" or
        type(baseline.get("evaluated_tasks")) is not int or
        baseline["evaluated_tasks"]<24 or
        not isinstance(baseline.get("held_out_dataset_sha256"),str) or
        len(baseline["held_out_dataset_sha256"])!=64 or
        any(c not in "0123456789abcdef" for c in baseline["held_out_dataset_sha256"])):
        raise ValueError("Real source-disjoint multi-domain >=24-case BASE model baseline required")
    return True

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source-private-train-dir",required=True)
    print(json.dumps(export(p.parse_args().source_private_train_dir),sort_keys=True))

"""H54 native ordered-drag image SFT, source-verified and opt-in remote only.

Authentic H53 original media remain in PRIVATE Kaggle. The PUBLIC GitHub repo
contains this conversion logic, hash receipts and fake first-party test data,
not external JPEG/JSON. This module NEVER trains a model.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from h53.dragon_train_shard import audit as audit_h53, verify_item
from training.artifacts import sha256_file, strict_json

SCHEMA="hala.h54.native-drag-sft.v1"
ORIGINAL_SHA256="9abab62624b45ead5db1ed0c1fdb5d7123f7c761c0a32b8a35f9ecfdebc96c09"
ORIGINAL_IMAGES=87
ORIGINAL_ACTIONS=760
FAMILY="hcompany_dragon_slider_train"
# A RESERVED family is not an actual existing held-out dataset. A genuine
# independently replayed baseline must be supplied before GPU training.
RESERVED_DEV_FAMILY="original_disjoint_drag_holdout_reserved"
SOURCE_RECEIPT=Path(__file__).resolve().parents[1]/"h53/source_evidence.json"
ROWS_NAME="h54_drag_sft.jsonl"
META_NAME="h54_drag_training_manifest.json"


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf8")


def verify_source(root, receipt_path=None):
    root=Path(root)
    receipt_path=Path(SOURCE_RECEIPT if receipt_path is None else receipt_path)
    official=audit_h53(root)
    external=strict_json(receipt_path.read_text("utf8"))
    manifest=strict_json((root/"manifest.json").read_text("utf8"))
    if (external.get("source_revision") != manifest.get("source_revision") or
        external.get("source_file") != "train/slider.tar" or
        external.get("jsonl_sha256") != ORIGINAL_SHA256 or
        manifest.get("jsonl_sha256") != ORIGINAL_SHA256 or
        official["source_jsonl_sha256"] != ORIGINAL_SHA256 or
        official["original_train_images"] != ORIGINAL_IMAGES or
        official["original_train_drag_annotations"] != ORIGINAL_ACTIONS or
        external.get("source_eval_used") is not False or
        manifest.get("source_eval_used") is not False or
        external.get("kaggle_dataset_id") != 12752110 or
        external.get("authenticated_private_listing_verified") is not True):
        raise ValueError("Frozen authentic H53 private TRAIN identity mismatch")
    expected=external.get("matched_source_sha256")
    incoming=manifest.get("matched_source_entries")
    if (not isinstance(expected,list) or len(expected)!=ORIGINAL_IMAGES or
        not isinstance(incoming,list) or len(incoming)!=ORIGINAL_IMAGES or
        any(set(item) != {"image_id","image_sha256","task_sha256","examples",
                           "source_image_range","source_task_range"} for item in expected)):
        raise ValueError("Original source per-image external byte SHA receipts required")
    lookup={e["image_id"]:e for e in expected}
    if len(lookup)!=ORIGINAL_IMAGES:raise ValueError("Original duplicate image identity")
    for actual in incoming:
        pinned=lookup.get(actual["image_id"])
        if pinned is None or pinned != actual:
            raise ValueError("Signed original H53 source image/task payload or provider TAR offset changed")
    return manifest,sha256_file(receipt_path)


def original_to_training_rows(root, receipt_path=None):
    root=Path(root)
    verify_source(root,receipt_path)
    original=[strict_json(row) for row in (root/"train_drag.jsonl").read_text("utf8").splitlines()]
    if len(original)!=ORIGINAL_ACTIONS:raise ValueError("Original 760 train source examples required")
    rows=[]
    for item in original:
        if (item.get("split")!="train" or item.get("domain")!="slider" or
            item.get("action_type")!="drag" or item.get("ordered") is not True or
            item.get("language")!="en" or
            item.get("image_sha256") is None or
            not isinstance(item.get("image_width"),int) or
            not isinstance(item.get("image_height"),int)):
            raise ValueError("Only exact original native TRAIN slider action type supported")
        action=item["completion"]
        if (not isinstance(action,dict) or
            set(action)!={"type","start_point","end_point","coordinate_space","ordered"} or
            action["type"]!="drag" or action["coordinate_space"]!="source_image_pixels" or
            action["ordered"] is not True):
            raise ValueError("Drag must remain a native ordered start/end action")
        goal=item["intent"]
        # The target text teaches exactly the action on the ORIGINAL image
        # coordinate system. Processor image resizing cannot silently relabel.
        prompt=(f"On this {item['image_width']}x{item['image_height']} original screenshot, "
           f"complete the GUI slider action: {goal}\n"
           "Return ONLY one JSON drag action with type, start_point, end_point, "
           "coordinate_space=source_image_pixels and ordered=true. "
           "The two points are integer ORIGINAL screenshot pixels. "
           "Do not click, navigate, execute, or use any external tool.")
        rows.append(dict(id=item["id"],split="train",family=FAMILY,language="en",
                 prompt=prompt,completion=json.dumps(action,sort_keys=True,
                    ensure_ascii=False,separators=(",",":")),
                 image_path=item["image_path"],image_sha256=item["image_sha256"]))
    return rows


def derived_artifacts(root,receipt_path=None):
    root=Path(root)
    source,receipt_hash=verify_source(root,receipt_path)
    rows=original_to_training_rows(root,receipt_path)
    raw=b"".join(json.dumps(r,ensure_ascii=False,sort_keys=True,
          separators=(",",":"),allow_nan=False).encode("utf8")+b"\n" for r in rows)
    photos=source["matched_source_entries"]
    media=sorted([dict(path="images/"+e["image_id"]+".jpg",sha256=e["image_sha256"],
                      family=FAMILY) for e in photos],key=lambda x:x["path"])
    manifest=dict(schema="hala.training.manifest.v1",source_schema=SCHEMA,
           split="train",rights_cleared=True,source_kind="rights_cleared_public",
           source_dataset="Hcompany/DragOn",source_revision=source["source_revision"],
           source_split="train/slider.tar",source_license="Apache-2.0",
           raw_original_jsonl_sha256=ORIGINAL_SHA256,
           source_receipt_sha256=receipt_hash,
           dataset_sha256=hashlib.sha256(raw).hexdigest(),token_budget=4096,
           families=[FAMILY],held_out_families=[RESERVED_DEV_FAMILY],
           independent_drag_dev_available=False,
           media=media,unique_source_images=ORIGINAL_IMAGES,train_rows=ORIGINAL_ACTIONS,
           native_action_type="drag",coordinate_space="source_image_pixels",
           no_benchmark_eval_source=True,training_executed=False,
           model_improvement_qualified=False)
    return raw,manifest,rows


def load_export(examples_path,manifest):
    examples_path=Path(examples_path)
    if (examples_path.name!=ROWS_NAME or examples_path.is_symlink() or
        not isinstance(manifest,dict)):
        raise ValueError("Only H54 source-bound original DragOn adapter candidate allowed")
    source=examples_path.parent
    raw,expected,rows=derived_artifacts(source)
    if manifest!=expected or examples_path.read_bytes()!=raw:
        raise ValueError("H54 original image/label bytes or derived training manifest changed")
    return rows


def export(root):
    root=Path(root)
    a=root/ROWS_NAME
    b=root/META_NAME
    if a.exists() or b.exists():
        raise FileExistsError("Never overwrite H54 original derived training evidence")
    raw,manifest,rows=derived_artifacts(root)
    with a.open("xb") as f:f.write(raw)
    try:
        with b.open("xb") as f:f.write(canonical(manifest))
    except BaseException:
        a.unlink()
        raise
    if load_export(a,manifest)!=rows:raise ValueError("H54 source replay differs")
    return dict(schema=SCHEMA,original_source_images=ORIGINAL_IMAGES,
        genuine_ordered_drag_labels=len(rows),dataset_sha256=manifest["dataset_sha256"],
        no_gpu_training=True,source_is_private_kaggle=True)


def validate_drag_baseline(baseline):
    if (not isinstance(baseline,dict) or
        baseline.get("evidence")!="real_inference" or
        baseline.get("task_kind")!="ordered_drag" or
        baseline.get("held_out_source")!="original_app_disjoint_drag_not_training" or
        not isinstance(baseline.get("held_out_dataset_sha256"),str) or
        len(baseline["held_out_dataset_sha256"])!=64 or
        any(c not in "0123456789abcdef" for c in baseline["held_out_dataset_sha256"]) or
        type(baseline.get("evaluated_tasks")) is not int or
        baseline["evaluated_tasks"]<16):
        raise ValueError("Actual independently sourced >=16-case native drag model baseline required")
    return True


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source-private-train-dir",required=True)
    args=p.parse_args()
    print(json.dumps(export(args.source_private_train_dir),sort_keys=True))

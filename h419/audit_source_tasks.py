"""H419 original source TASK family leakage audit, separate from image SHA.

The native model uses original H60 drag actions. A source image may have
several raw task annotations, and two different JPEGs could be derived from
one underlying source task. This gate checks BOTH the source image SHA and
the original source_task_sha256 against two untouched evaluation sets.

No GPU, no model scoring, no public dataset export. Owner private only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from h419.kaggle import h419_native_drag_continuation as h

TOTAL_ORIGINAL_ACTIONS=500554
TOTAL_ORIGINAL_IMAGES=47037
EXPECTED={
    "h419_train_original":30000,
    "h417_old_dev_original":96,
    "h419_new_dev_original":96,
}
LABELSETS=tuple(EXPECTED)


def source_paths(root):
    root=Path(root)
    paths={name:root/"h415"/name for name,_ in h.SOURCE_SHAS.values()}
    paths.update({name:root/"h418"/name for name in h.H419_EXTRA_SHAS})
    paths.update({
        h.BIL_FILE:root/"h415"/h.BIL_FILE,
        h.CURRIC_FILE:root/"h415"/h.CURRIC_FILE,
        h.H418_ROWS:root/"new-30k"/h.H418_ROWS,
        h.H418_RECEIPT_FILE:root/"new-30k"/h.H418_RECEIPT_FILE,
        h.PARENT_ADAPTER_FILE:root/"parent"/h.PARENT_ADAPTER_FILE,
        h.PARENT_RECEIPT_FILE:root/"parent"/h.PARENT_RECEIPT_FILE,
        "h417_original_receipt.sha256":root/"parent"/"h417_original_receipt.sha256"
    })
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("H419 all nine original ZIPs, source receipt and private parent required")
    return paths


def prove_pairwise_isolated(images,tasks):
    """Pure safety contract; test forged same-task/different-image leakage."""
    if (set(images)!=set(LABELSETS) or set(tasks)!=set(LABELSETS) or
        any(type(images[k]) is not set or type(tasks[k]) is not set
            for k in LABELSETS)):
        raise ValueError("H419 original source-task partition keys missing")
    if any(len(images[k])!=EXPECTED[k] for k in LABELSETS):
        raise ValueError("H419 frozen old/new source image split changed")
    if any(not tasks[k] for k in LABELSETS):
        raise ValueError("H419 some original training/development source tasks missing")
    overlaps={}
    for i,a in enumerate(LABELSETS):
        for b in LABELSETS[i+1:]:
            pix=images[a]&images[b]
            raw=tasks[a]&tasks[b]
            overlaps[a+" vs "+b]={"image_sha_overlap":len(pix),
                                  "original_source_task_sha_overlap":len(raw)}
            if pix or raw:
                raise ValueError("H419 source-image or underlying-task training/eval leakage")
    return overlaps


def audit(root):
    owner=source_paths(root)
    original=h.only_input_file
    zips=None
    extra=[]
    try:
        h.only_input_file=lambda name:owner[name]
        zips,extra,train,dev,order,old,fresh=h.h419_prepare_sources()
        groups={
          "h419_train_original":{r["image_sha256"] for r in train
                                  if r["kind"]=="original"},
          "h417_old_dev_original":{r["image_sha256"] for r in old
                                   if r["kind"]=="original"},
          "h419_new_dev_original":{r["image_sha256"] for r in fresh},
        }
        if (len(train),len(dev),len(order),len(old),len(fresh))!=(
                30976,240,37808,144,96):
            raise ValueError("H419 actual 30K/240 source partition changed")
        all_images=set()
        target=set().union(*groups.values())
        tasks={name:set() for name in groups}
        annotation_counts=Counter()
        image_annotation_task={k:Counter() for k in groups}
        scanned=0
        for name,z in zips["_by_file"].items():
            manifest=h.strict_load(z.read("manifest.json").decode("utf8"))
            domain=manifest["source_domain"]
            with z.open("labels.jsonl") as file:
                for line in file:
                    if not line.strip():continue
                    scanned+=1
                    row=h.strict_load(line.decode("utf8"))
                    image_sha=row["image_sha256"]
                    all_images.add(image_sha)
                    if image_sha not in target:continue
                    task=row.get("source_task_sha256")
                    if (type(task) is not str or len(task)!=64 or
                        any(c not in "0123456789abcdef" for c in task) or
                        row.get("domain")!=domain or row.get("split")!="train"):
                        raise ValueError("H419 original H60 task annotation invalid")
                    for label,images in groups.items():
                        if image_sha in images:
                            tasks[label].add(task)
                            annotation_counts[label]+=1
                            image_annotation_task[label][image_sha]+=1
        if (scanned!=TOTAL_ORIGINAL_ACTIONS or
            len(all_images)!=TOTAL_ORIGINAL_IMAGES or
            any(len(image_annotation_task[k])!=EXPECTED[k] for k in groups)):
            raise ValueError("H419 original source task census or original media incomplete")
        overlap=prove_pairwise_isolated(groups,tasks)
        return {
            "schema":"hala.h419.private-original-source-task-isolation.v1",
            "source":"Hcompany/DragOn original TRAIN-only H60 private nine SHA-pinned ZIPs",
            "source_revision":h.SOURCE_REV,
            "original_h60_source_actions_scanned":scanned,
            "original_source_images_with_train_annotations":len(all_images),
            "train_new_original_screen_sha":len(groups["h419_train_original"]),
            "old_dev_original_screen_sha":len(groups["h417_old_dev_original"]),
            "fresh_dev_original_screen_sha":len(groups["h419_new_dev_original"]),
            "original_task_sha_counts":{k:len(v) for k,v in tasks.items()},
            "raw_source_action_records_for_selected_images":dict(annotation_counts),
            "overlaps":overlap,
            "h417_heldout_used_for_training":False,
            "source_only_author_visible_holdout":True,
            "independent_external_ood":False,
            "model_training_performed_by_this_audit":False,
            "model_release_qualified":False,
        }
    finally:
        h.only_input_file=original
        for z in extra:z.close()
        if zips:
            for obj in zips.values():
                if isinstance(obj,zipfile.ZipFile):obj.close()


def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--owner-sources",required=True)
    p.add_argument("--receipt-root",required=True)
    args=p.parse_args()
    destination=Path(args.receipt_root)
    destination.mkdir(parents=True,exist_ok=True)
    receipt=destination/"receipt.json"
    sidecar=destination/"receipt.sha256"
    if receipt.exists() or sidecar.exists():
        raise FileExistsError("H419 original source-task receipt create-only")
    result=audit(args.owner_sources)
    raw=json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False,
                   allow_nan=False).encode("utf8")
    try:
        with receipt.open("xb") as f:f.write(raw)
        with sidecar.open("x",encoding="ascii") as f:
            f.write(sha256(raw).hexdigest()+"\n")
    except BaseException:
        receipt.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    print("H419_SOURCE_TASK_DISJOINT_PROVED",
          json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    cli()

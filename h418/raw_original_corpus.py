"""H418: source-original, SHApinned, H417-train-and-dev disjoint native DRAG.

This is the preferred large-scale source route when historical H61 has
re-encoded legacy frames. Reuse the NINE originally trained, locally
SHA-verified H60 TRAIN ZIPs and select an additional 30K truly new,
original JPEG image-level examples with genuine native drag supervision.

NEVER a public release / real GUI authority / independently held OOD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

from h417.kaggle import h417_native_drag_lora as parent
from h418 import curate as legal

TARGET=30000
MIN_AVAILABLE_NEW=35000
SOURCE_IMAGE_UNIQUE=47037
SOURCE_ACTIONS=500554
SEED="h418-original-30k-new-images-v1"
OUT_ROWS="h418_fresh_original_30k_drag.jsonl"
OUT_RECEIPT="h418_fresh_original_30k_manifest.json"


def sha_hex(x:bytes)->str:
    return sha256(x).hexdigest()


def h417_train_and_dev_sets(root):
    """Regenerate the real source-image partition, not hand-edited hashes."""
    root=Path(root)
    former=parent.only_input_file
    opened=None
    try:
        parent.only_input_file=lambda name:root/name
        opened,raw,curriculum,bilingual=parent.get_research_sources()
        train,dev,order=parent.partition(raw,curriculum,bilingual)
        train_sha={r["image_sha256"] for r in train}
        dev_sha={r["image_sha256"] for r in dev}
        if (len(train)!=11952 or len(dev)!=144 or len(order)!=14880 or
            len(train_sha)!=11952 or len(dev_sha)!=144 or
            train_sha&dev_sha):
            raise ValueError("H418 H417 frozen TRAIN+DEV original image boundaries changed")
        return train_sha,dev_sha
    finally:
        parent.only_input_file=former
        if opened:
            for bundle in opened.values():bundle.close()


def original_record(row, domain, filename, source_sha):
    if (type(row) is not dict or row.get("split")!="train" or
        row.get("domain")!=domain or
        row.get("image_path") is None or
        type(row.get("id")) is not str or
        len(row["id"])<4 or
        row.get("image_sha256")!=source_sha or
        type(row.get("image_width")) is not int or
        type(row.get("image_height")) is not int or
        row["image_width"]<=0 or row["image_height"]<=0 or
        type(row.get("ordered")) is not bool or
        type(row.get("intent")) is not str or
        not row["intent"].strip() or
        len(row["intent"])>2000):
        raise ValueError("H418 H60 original TRAIN label provenance or native instruction altered")
    completion=row.get("completion")
    if (type(completion) is not dict or
        completion.get("type")!="drag" or
        completion.get("coordinate_space")!="source_image_pixels" or
        completion.get("ordered") is not row["ordered"]):
        raise ValueError("H418 authentic H60 native ordered DRAG completion changed")
    for field in ("start_point","end_point"):
        xy=completion.get(field)
        if (type(xy) is not list or len(xy)!=2 or
            any(type(v) is not int for v in xy) or
            not 0<=xy[0]<=row["image_width"] or
            not 0<=xy[1]<=row["image_height"]):
            raise ValueError("H418 original drag endpoint outside H60 JPEG source dimensions")
    for prop in ("image_path","source_task_sha256"):
        if type(row.get(prop)) is not str or not row[prop]:
            raise ValueError("H418 original H60 media or task path missing")
    return {
       "schema":"hala.h418.original-native-drag-train.v1",
       "id":row["id"],
       "source_domain":domain,
       "source_zip":filename,
       "source_zip_sha256":legal.ALL_SHAS[filename],
       "split":"train",
       "language":row["language"],
       "intent":row["intent"],
       "image_path":row["image_path"],
       "image_sha256":source_sha,
       "image_width":row["image_width"],
       "image_height":row["image_height"],
       "original_source_task_sha256":row["source_task_sha256"],
       "ordered":row["ordered"],
       "native_drag_completion":completion,
    }


def inventory(source,archives,already_train,heldout):
    """One original native task per original JPEG SHA, new to H417."""
    if (type(source) is not dict or
        type(already_train) is not set or len(already_train)!=11952 or
        type(heldout) is not set or len(heldout)!=144 or
        already_train&heldout):
        raise ValueError("H418 source index or H417 exact train/DEV holdout missing")
    chosen={}
    represented=Counter()
    original_source_hashes=set()
    total_actions=0
    excluded=Counter()
    for z in archives:
        manifest=parent.strict_load(z.read("manifest.json").decode("utf8"))
        domain=manifest["source_domain"]
        filename=next((name for name,pin in legal.ALL_SHAS.items()
                       if z.filename and Path(z.filename).name==name),None)
        if filename is None:
            raise ValueError("H418 original ZIP source name unknown")
        for raw_line in z.read("labels.jsonl").decode("utf8").splitlines():
            if not raw_line.strip():continue
            total_actions+=1
            row=parent.strict_load(raw_line)
            image_sha=row.get("image_sha256")
            source_path=row.get("image_path")
            pair=source.get((domain,source_path))
            if (pair is None or len(pair)!=3 or
                pair[1]!=filename or pair[2]!=image_sha):
                raise ValueError("H418 original H60 labelled TRAIN JPEG not pinned to source ZIP")
            original_source_hashes.add(image_sha)
            represented[domain]+=1
            if image_sha in already_train:
                excluded["h417_prior_train_source_sha"]+=1
                continue
            if image_sha in heldout:
                excluded["h417_dev_original_sha"]+=1
                continue
            rep=original_record(row,domain,filename,image_sha)
            key=sha_hex((SEED+"|"+rep["id"]).encode("utf8"))
            existing=chosen.get(image_sha)
            if existing is None or key<existing[0]:
                chosen[image_sha]=(key,rep)
    if total_actions!=SOURCE_ACTIONS or len(original_source_hashes)!=SOURCE_IMAGE_UNIQUE:
        raise ValueError("H418 full original 500554 actions / 47037 source images census drift")
    # Globally unique original SHA count includes H417 already-trained images.
    new_image_sha={sha for sha in chosen}
    if (len(new_image_sha)<MIN_AVAILABLE_NEW or
        new_image_sha&already_train or
        new_image_sha&heldout):
        raise ValueError("H418 insufficient genuine NEW source-original images")
    result={k:[] for k in parent.DOMAINS}
    for _,rec in chosen.values():
        result[rec["source_domain"]].append(rec)
    for domain in result:
        result[domain].sort(key=lambda r:sha_hex(
            (SEED+"|"+r["image_sha256"]).encode("utf8")))
    return result, dict(represented),dict(excluded)


def balanced_select(grouped,source,target=TARGET):
    if type(target) is not int or target<=0:
        raise ValueError("H418 original source target must be positive")
    selection=[]
    taken=set()
    position=0
    while len(selection)<target:
        progress=False
        for domain in parent.DOMAINS:
            bucket=grouped.get(domain)
            if type(bucket) is not list:
                raise ValueError("H418 missing source domain")
            if position>=len(bucket):continue
            case=bucket[position]
            key=case["image_sha256"]
            if key in taken:
                raise ValueError("H418 repeated source image SHA after global dedup")
            pair=source.get((domain,case["image_path"]))
            if pair is None or pair[1]!=case["source_zip"] or pair[2]!=key:
                raise ValueError("H418 source ZIP has drifted before actual media proof")
            selection.append(case)
            taken.add(key)
            progress=True
            if len(selection)==target:break
        if not progress:break
        position+=1
    if len(selection)!=target:
        raise ValueError("H418 not enough original new images to fill native drag budget")
    if len(taken)!=target:
        raise ValueError("H418 image-level original training repetition")
    return selection


def prove_selected_bytes(selection,source):
    from PIL import Image
    actual=[]
    source_count=Counter()
    for i,case in enumerate(selection):
        z,name,raw_sha=source[(case["source_domain"],case["image_path"])]
        buf=z.read(case["image_path"])
        if sha_hex(buf)!=case["image_sha256"] or raw_sha!=case["image_sha256"]:
            raise ValueError("H418 selected actual original TRAIN JPEG bytes SHA mismatch")
        if not buf.startswith(b"\xff\xd8\xff"):
            raise ValueError("H418 selected original JPEG media invalid")
        with Image.open(BytesIO(buf)) as frame:
            if frame.format!="JPEG" or frame.size!=(
                case["image_width"],case["image_height"]):
                raise ValueError("H418 original native point/source image geometry mismatch")
            frame.verify()
        source_count[case["source_domain"]]+=1
        actual.append(case)
        if (i+1)%2000==0:
            print("H418_ORIGINAL_JPEG_SHA_PROOF",i+1,"/",len(selection),flush=True)
    return actual,dict(source_count)


def export(actual,domain_counts,represented,excluded,destination,
           train_sha,dev_sha):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    output=destination/OUT_ROWS
    receipt_file=destination/OUT_RECEIPT
    if output.exists() or receipt_file.exists():
        raise FileExistsError("H418 authentic source output CREATE ONLY")
    ids={r["id"] for r in actual}
    image_sha={r["image_sha256"] for r in actual}
    if (len(actual)!=TARGET or len(ids)!=TARGET or len(image_sha)!=TARGET or
        image_sha&train_sha or image_sha&dev_sha):
        raise ValueError("H418 true original TRAIN/source or H417 holdout overlap")
    blob=b"".join(legal.canonical(r)+b"\n" for r in actual)
    rec={
        "schema":"hala.h418.owner-private-30k-fresh-original-drags.v1",
        "status":"ORIGINAL_SOURCE_BYTES_VERIFIED_MODEL_NOT_TRAINED",
        "source_revision":legal.SOURCE_REV,
        "full_upstream_tar_sha_verified":False,
        "source_split":"train",
        "original_native_action_rows_available":SOURCE_ACTIONS,
        "original_source_labelled_unique_sha":SOURCE_IMAGE_UNIQUE,
        "h417_train_image_sha_disjoint":True,
        "h417_dev144_image_sha_disjoint":True,
        "selected_unique_original_train_images":len(actual),
        "selected_training_rows":len(actual),
        "selected_bytes_sha256":sha_hex(blob),
        "original_9_archive_sha256":legal.ALL_SHAS,
        "new_images_by_domain":domain_counts,
        "original_labelled_domain_action_counts":represented,
        "h417_exclusion_counts":excluded,
        "actual_all_30000_original_jpeg_bytes_sha_verified":True,
        "model_training_executed":False,
        "original_gui_source_rights_independently_approved":False,
        "external_independent_ood":False,
        "weights_public":False,
        "model_release_qualified":False,
    }
    try:
        with output.open("xb") as o:o.write(blob)
        with receipt_file.open("xb") as o:o.write(
            json.dumps(rec,sort_keys=True,indent=2,allow_nan=False).encode())
    except BaseException:
        output.unlink(missing_ok=True);receipt_file.unlink(missing_ok=True)
        raise
    return rec


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--h415-zips",required=True)
    p.add_argument("--h418-zips",required=True)
    p.add_argument("--h417-originals",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    index,opened=legal.open_archives(args.h415_zips,args.h418_zips)
    try:
        train,dev=h417_train_and_dev_sets(args.h417_originals)
        groups,represented,excluded=inventory(index,opened,train,dev)
        selected=balanced_select(groups,index)
        verified,domain_counts=prove_selected_bytes(selected,index)
        state=export(verified,domain_counts,represented,excluded,
                     args.out,train,dev)
        print("H418_REAL_NEW_30000_ORIGINAL_DRAG_SOURCE",
              json.dumps(state,sort_keys=True),flush=True)
    finally:
        for z in opened:z.close()


if __name__=="__main__":
    run()

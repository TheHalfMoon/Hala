"""H422 genuine H60 original SAME-JPEG two-instruction native-DRAG curriculum.

No synthetic fabricated drag points: select 16,000 ORIGINAL source TRAIN
JPEGs, with first H418 native action plus true different H420 source
action for SAME original raw image SHA, DIFFERENT instruction and source
geometry. This yields 32,000 original image-conditioned action labels
while reserving all H417/H419 DEV and H421 counterfactual unseen pairs.

The exact source H60 image bytes of EACH selected original JPEG are
reopened and SHA256 verified, original source task provenance pinned.
Additional AR original browser TRAIN 976 screenshots are separate from
this EN-original paired-data source, added only at actual GPU training.

SOURCE-only output; no model training or production approval.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from h418 import curate as h418
from h420 import curate_alternative_drags as h420
from h420 import counterfactual_score as h421

SEED="hala-h422-sixteen-thousand-same-screen-two-authentic-drags-v1"
ORIGINAL_PAIR_IMAGES=16000
ORIGINAL_TRAIN_ACTIONS=32000
PER_DOMAIN={"slider":5333,"slide_resize":5333,"text_highlight":5334}
H420_SECOND_ROWS_SHA="58c88bf09c047973675ca6dd906eae8d6832005f5da338d492a67c8dd35b91ed"
H420_SECOND_RECEIPT_SHA="ae32019e4ca81213cab3cc4e7135a2b09b68ee54e809dd0a553eacad14c51526"
H421_UNSEEN_ROWS_SHA="42d93219ec4083920462d4194b175154f1f12d9b3e7b36ffd1068d8f8a4d96b9"
H421_UNSEEN_RECEIPT_SHA="02a4d17ead71fc06019a83669a04c9779430de5a7a48f800429ad2241d2ebaf4"
ROWS_FILE="h422_genuine_original_paired_32k_actions.jsonl"
RECEIPT_FILE="h422_original_paired_source_receipt.json"
SCHEMA="hala.h422.original-same-screen-paired-native-drag-train.v1"


def sha(path):
    h=sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda:stream.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def verified_manifest(path,digest,schema,**terms):
    if sha(path)!=digest:
        raise ValueError("H422 pinned original source receipt SHA invalid")
    rec=h418.parent.strict_load(Path(path).read_text("utf8"))
    if rec.get("schema")!=schema:
        raise ValueError("H422 original source receipt schema drift")
    for field,expected in terms.items():
        if rec.get(field)!=expected:
            raise ValueError("H422 original source receipt provenance mismatch: "+field)
    return rec


def native_signature(row):
    if type(row) is not dict or row.get("type")!="drag":
        raise ValueError("H422 requires original two-point DRAG only")
    if row.get("coordinate_space")!="source_image_pixels" or type(row.get("ordered")) is not bool:
        raise ValueError("H422 unqualified native coordinate space or ordered")
    a,b=row.get("start_point"),row.get("end_point")
    if (type(a) is not list or type(b) is not list or
        len(a)!=2 or len(b)!=2 or
        any(type(v) is not int or v<0 for v in a+b)):
        raise ValueError("H422 corrupt original native source coordinates")
    return tuple(a),tuple(b),row["ordered"]


def prove_inputs(first_rows,first_receipt,second_rows,second_receipt,
                 third_rows,third_receipt,source_task_receipt):
    baseline=h420.source_manifest(first_rows,first_receipt,source_task_receipt)
    second_proof=verified_manifest(
         second_receipt,H420_SECOND_RECEIPT_SHA,
         "hala.h420.private-authentic-second-native-drags.v1",
         status="ORIGINAL_COMPLEMENTARY_SOURCE_BYTES_VERIFIED_MODEL_NOT_TRAINED",
         new_original_distinct_complementary_native_actions=29827,
         original_source_second_actions_jsonl_sha256=H420_SECOND_ROWS_SHA,
         original_first_30000_source_sha256=h420.H418_ROWS_SHA,
         original_first_30000_source_receipt_sha256=h420.H418_RECEIPT_SHA,
         original_task_heldout_isolation_receipt_sha256=h420.H419_SOURCE_TASK_AUDIT_SHA,
         all_selected_original_JPEG_bytes_SHA_verified=True,
         model_training_executed=False,
         model_release_qualified=False,
         external_independent_ood=False)
    if sha(second_rows)!=H420_SECOND_ROWS_SHA:
        raise ValueError("H422 H420 second original task corpus byte SHA mismatch")
    third_proof=verified_manifest(
         third_receipt,H421_UNSEEN_RECEIPT_SHA,
         "hala.h421.source-original-counterfactual-dev.v1",
         owner_heldout_source_image_pairs=192,
         native_two_instruction_eval_cases=384,
         frozen_original_dev_rows_sha256=H421_UNSEEN_ROWS_SHA,
         excluded_h417_and_h419_train_and_dev_source_images=True,
         excluded_source_task_sha_of_h417_h419_models_and_dev=True,
         model_training_executed=False,
         model_evaluation_executed=False,
         independent_external_ood=False)
    if sha(third_rows)!=H421_UNSEEN_ROWS_SHA:
        raise ValueError("H422 authentic untouched 384-case source eval SHA drift")
    reserved={h418.parent.strict_load(line)["image_sha256"]
              for line in Path(third_rows).read_text("utf8").splitlines() if line.strip()}
    if len(reserved)!=192 or reserved&set(baseline):
        raise ValueError("H422 original H421 unseen evaluation images entered TRAIN")
    duplicate=set()
    pairs={}
    for line in Path(second_rows).open("r",encoding="utf8"):
        if not line.strip():continue
        second=h418.parent.strict_load(line)
        image_sha=second.get("image_sha256")
        old=baseline.get(image_sha)
        if (old is None or image_sha in duplicate or image_sha in reserved
            or second.get("schema")!="hala.h420.original-second-native-drag-train.v1"
            or second.get("split")!="train"
            or second.get("same_verified_image_as_h419_train") is not True
            or second.get("no_original_h417_or_h419_dev_image_or_task") is not True
            or second.get("source_zip")!=old["source_zip"]
            or second.get("source_zip_sha256")!=old["source_zip_sha256"]
            or second.get("source_domain")!=old["source_domain"]
            or second.get("h419_first_action_id")!=old["id"]
            or second.get("image_width")!=old["image_width"]
            or second.get("image_height")!=old["image_height"]
            or second.get("language")!="en"
            or second.get("intent")==old["intent"]
            or second.get("id")==old["id"]
            or second.get("h419_first_native_action_sha256")!=sha256(
                 h418.canonical(old["native_drag_completion"])).hexdigest()):
            raise ValueError("H422 paired source first/second instruction contamination")
        primary=native_signature(old["native_drag_completion"])
        alternative=native_signature(second["native_drag_completion"])
        for native in (primary,alternative):
            for x,y in native[:2]:
                if x>old["image_width"] or y>old["image_height"]:
                    raise ValueError("H422 original pixel point outside authentic JPEG dimensions")
        if primary[:2]==alternative[:2]:
            raise ValueError("H422 second native target is just a duplicate first action")
        shift=sum(abs(x-y) for before,after in zip(primary[:2],alternative[:2])
                  for x,y in zip(before,after))
        separation=shift/(old["image_width"]+old["image_height"])
        if (abs(separation-second["additional_action_ratio_of_image_dims"])>0.000002
            or separation<=0 or
            second.get("different_instruction_text") is not True):
            raise ValueError("H422 misleading contrastive native-source geometry")
        pairs[image_sha]=(old,second)
        duplicate.add(image_sha)
    if len(pairs)!=29827:
        raise ValueError("H422 original 29827 H420 paired source images missing")
    return pairs,reserved


def select(pairs):
    families={domain:[] for domain in PER_DOMAIN}
    for image_sha,(first,second) in pairs.items():
        domain=first["source_domain"]
        if domain not in families:
            raise ValueError("H422 unknown original source action domain")
        rank=sha256((SEED+"|"+image_sha).encode("utf8")).hexdigest()
        families[domain].append((rank,image_sha,first,second))
    selected=[]
    for domain in PER_DOMAIN:
        family=sorted(families[domain],key=lambda x:x[0])
        quota=PER_DOMAIN[domain]
        if len(family)<quota:
            raise ValueError("H422 not enough source-original paired image-action examples")
        selected.extend(family[:quota])
    selected.sort(key=lambda x:x[0])
    if (len(selected)!=ORIGINAL_PAIR_IMAGES or
        len({x[1] for x in selected})!=ORIGINAL_PAIR_IMAGES):
        raise ValueError("H422 genuine original training image SHA dedup failed")
    return selected


def proof_original_bytes(selected,archive_index):
    from PIL import Image
    checked=0
    for rank,image_sha,first,second in selected:
        for row in (first,second):
            key=(row["source_domain"],row["image_path"])
            archive=archive_index.get(key)
            if (archive is None or archive[1]!=row["source_zip"]
                or archive[2]!=image_sha):
                raise ValueError("H422 source image is not a labelled original pinned ZIP member")
            raw=archive[0].read(row["image_path"])
            if sha256(raw).hexdigest()!=image_sha or raw[:3]!=b"\xff\xd8\xff":
                raise ValueError("H422 original paired JPEG bytes SHA mismatch")
            with Image.open(BytesIO(raw)) as im:
                if (im.format!="JPEG" or im.size!=(
                    row["image_width"],row["image_height"])):
                    raise ValueError("H422 original JPEG dimensions not native image pixel geometry")
                im.verify()
            checked+=1
        if checked%4000==0:
            print("H422_GENUINE_ORIGINAL_DRAG_SOURCE_MEDIA",checked,"/",ORIGINAL_TRAIN_ACTIONS,
                  flush=True)
    if checked!=ORIGINAL_TRAIN_ACTIONS:
        raise ValueError("H422 source first/second JPEG proof incomplete")
    return checked


def output_selection(selected):
    lines=[]
    ids=set()
    for rank,image_sha,first,second in selected:
        for idx,record in enumerate((first,second)):
            label=record["native_drag_completion"]
            if record["id"] in ids:
                raise ValueError("H422 duplicate source-original task ID")
            ids.add(record["id"])
            row={
               "schema":SCHEMA,"pair_id":image_sha,"alternative_index":idx,
               "id":record["id"],"original_source_task_sha256":
                   record["original_source_task_sha256"],
               "split":"train","language":"en",
               "source_domain":record["source_domain"],
               "source_zip":record["source_zip"],
               "source_zip_sha256":record["source_zip_sha256"],
               "image_path":record["image_path"],
               "image_sha256":image_sha,
               "image_width":record["image_width"],
               "image_height":record["image_height"],
               "instruction":record["intent"],
               "native_drag_completion":label,
               "original_source_JPEG_bytes_SHA_verified":True,
               "H421_unseen_source_image_excluded":True,
               "H417_H419_dev_source_image_task_excluded":True,
            }
            lines.append(row)
    if len(lines)!=ORIGINAL_TRAIN_ACTIONS or len(ids)!=ORIGINAL_TRAIN_ACTIONS:
        raise ValueError("H422 original first/second source task identity not unique")
    return lines


def export(rows,root,verified_count):
    if (type(rows) is not list or len(rows)!=ORIGINAL_TRAIN_ACTIONS
        or verified_count!=ORIGINAL_TRAIN_ACTIONS
        or len({r["id"] for r in rows})!=ORIGINAL_TRAIN_ACTIONS
        or len({r["image_sha256"] for r in rows})!=ORIGINAL_PAIR_IMAGES):
        raise ValueError("H422 genuine 16K original JPEG /32K native action proof incomplete")
    role_counts=Counter(r["alternative_index"] for r in rows)
    if role_counts!={0:ORIGINAL_PAIR_IMAGES,1:ORIGINAL_PAIR_IMAGES}:
        raise ValueError("H422 all 16K original TRAIN screens need two real instructions")
    observed={}
    for r in rows:
        if (r.get("schema")!=SCHEMA or r.get("split")!="train" or
            r.get("original_source_JPEG_bytes_SHA_verified") is not True or
            r.get("H421_unseen_source_image_excluded") is not True or
            r.get("H417_H419_dev_source_image_task_excluded") is not True):
            raise ValueError("H422 output image provenance / reserved DEV not qualified")
        pair=observed.setdefault(r["image_sha256"],{})
        if r["alternative_index"] in pair:
            raise ValueError("H422 repeated native action for one same-JPEG source")
        pair[r["alternative_index"]]=r
    for image,entries in observed.items():
        if set(entries)!={0,1}:
            raise ValueError("H422 original same image has incomplete two-prompt pair")
        one,two=entries[0],entries[1]
        if (one["instruction"]==two["instruction"] or
            native_signature(one["native_drag_completion"])[:2]==
            native_signature(two["native_drag_completion"])[:2] or
            one["source_zip"]!=two["source_zip"] or
            one["source_domain"]!=two["source_domain"] or
            one["image_width"]!=two["image_width"] or
            one["image_height"]!=two["image_height"]):
            raise ValueError("H422 original two tasks not instruction/target paired")
    folder=Path(root);folder.mkdir(parents=True,exist_ok=True)
    target=folder/ROWS_FILE
    receipt_file=folder/RECEIPT_FILE
    sidecar=folder/"receipt.sha256"
    if target.exists() or receipt_file.exists() or sidecar.exists():
        raise FileExistsError("H422 actual source training rows CREATE ONLY")
    blob=b"".join(h418.canonical(r)+b"\n" for r in rows)
    counts=Counter(r["source_domain"] for r in rows if r["alternative_index"]==0)
    report={
       "schema":"hala.h422.owner-private-original-16k-paired-native-drag-train.v1",
       "status":"SOURCE_VERIFIED_32000_PAIRED_NATIVE_DRAGS_MODEL_NOT_TRAINED",
       "source_split":"train",
       "genuine_unique_original_JPEG_train":ORIGINAL_PAIR_IMAGES,
       "genuine_original_image_conditional_action_labels":ORIGINAL_TRAIN_ACTIONS,
       "distinct_original_instructions_per_image":2,
       "genuine_paired_original_JPEG_media_reads":verified_count,
       "each_original_image_sha_belongs_to_one_training_pair":True,
       "real_original_media_SHA_replayed":True,
       "original_train_by_domain":dict(counts),
       "original_training_action_rows_sha256":sha256(blob).hexdigest(),
       "first_30000_original_image_rows_sha256":h420.H418_ROWS_SHA,
       "first_30000_original_source_receipt_sha256":h420.H418_RECEIPT_SHA,
       "second_29827_original_action_rows_sha256":H420_SECOND_ROWS_SHA,
       "second_29827_original_source_receipt_sha256":H420_SECOND_RECEIPT_SHA,
       "untouched_H421_eval_384_source_rows_sha256":H421_UNSEEN_ROWS_SHA,
       "untouched_H421_eval_192_original_screens_excluded":True,
       "untouched_H417_H419_dev_and_source_task_excluded":True,
       "original_H60_nine_ZIP_sha256":h418.ALL_SHAS,
       "source_risk_external_rights_independently_verified":False,
       "independent_external_ood":False,
       "model_training_executed":False,
       "model_release_qualified":False,
       "weights_public":False,
    }
    try:
        with target.open("xb") as f:f.write(blob)
        b=json.dumps(report,sort_keys=True,indent=2,allow_nan=False).encode()
        with receipt_file.open("xb") as f:f.write(b)
        with sidecar.open("x",encoding="ascii",newline="\n") as f:
            f.write(sha256(b).hexdigest())
    except BaseException:
        target.unlink(missing_ok=True);receipt_file.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return report


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h418-rows",required=True)
    ap.add_argument("--h418-receipt",required=True)
    ap.add_argument("--h419-task-receipt",required=True)
    ap.add_argument("--h420-rows",required=True)
    ap.add_argument("--h420-receipt",required=True)
    ap.add_argument("--h421-rows",required=True)
    ap.add_argument("--h421-receipt",required=True)
    ap.add_argument("--h415-zips",required=True)
    ap.add_argument("--h418-zips",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    pairs,reserved=prove_inputs(
        args.h418_rows,args.h418_receipt,args.h420_rows,args.h420_receipt,
        args.h421_rows,args.h421_receipt,args.h419_task_receipt)
    selected=select(pairs)
    index,zips=h418.open_archives(args.h415_zips,args.h418_zips)
    try:
        count=proof_original_bytes(selected,index)
        rows=output_selection(selected)
        print("H422_REAL_TWO_TARGET_TRAIN_SOURCE",
              json.dumps(export(rows,args.out,count),sort_keys=True),flush=True)
    finally:
        for zipfile in zips:zipfile.close()


if __name__=="__main__":
    run()

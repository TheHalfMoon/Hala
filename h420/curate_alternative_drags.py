"""H420 source-original complementary NATIVE drag actions for H419 training images.

H418 qualified 30,000 distinct original H60 TRAIN JPEGs with one action
each. Original H60 source labels contain *additional genuinely different*
ordered/unordered start/end DRAG tasks for many of those same images.

This NEW curation adds at most ONE different original action PER H418
TRAIN image, never a held-out H417/H419 evaluation source image or task.
Actual media original JPEG SHA is replayed for every admitted action.
Output is owner PRIVATE, CREATE ONLY, NO model training or release claims.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

from h418 import curate as source
from h418 import raw_original_corpus as raw

H418_ROWS_SHA="e7fce082d6ccd931515beacbc6cfc2455a6683c7c92566a245fda9f2182a1605"
H418_RECEIPT_SHA="b7b75d6aa2587c3eba83a3934fe4cdf3cacd48a5b970b5a60a640c89ee3054cf"
ORIGINAL_IMAGE_N=30000
MIN_DISTINCT_ALTERNATIVES=29000
FROZEN_CANDIDATE_N=29827
FROZEN_COUNT_BY_DOMAIN={
    "slider":9914,
    "slide_resize":9913,
    "text_highlight":10000,
}
ORIGINAL_NATIVE_ACTIONS=500554
ORIGINAL_SOURCE_IMAGES=47037
H419_SOURCE_TASK_AUDIT_SHA="fff2cc485424f5263780cd234dcb93d302b2f5cb421206875f49d8df7dd6be3b"
SOURCE_SPLIT="train"
SEED="hala-h420-second-authentic-original-task"
OUTPUT="h420_second_original_native_drag.jsonl"
RECEIPT="h420_original_second_action_receipt.json"


def _hex_sha(x):
    return (type(x) is str and len(x)==64 and
            all(c in "0123456789abcdef" for c in x))


def native_tuple(target):
    if (type(target) is not dict or target.get("type")!="drag" or
        target.get("coordinate_space")!="source_image_pixels" or
        type(target.get("ordered")) is not bool):
        raise ValueError("H420 source action not an original native DRAG")
    a=target.get("start_point")
    b=target.get("end_point")
    if (type(a) is not list or type(b) is not list or
        len(a)!=2 or len(b)!=2 or
        any(type(v) is not int for v in a+b)):
        raise ValueError("H420 source action native endpoints invalid")
    return (tuple(a),tuple(b),target["ordered"])


def source_manifest(rows_path,receipt_path,source_task_receipt):
    rows_path=Path(rows_path);receipt_path=Path(receipt_path)
    if source.parent.sha_file(rows_path)!=H418_ROWS_SHA:
        raise ValueError("H420 original H418 30K TRUE source selection SHA mismatch")
    if source.parent.sha_file(receipt_path)!=H418_RECEIPT_SHA:
        raise ValueError("H420 prior original H418 byte-proven receipt SHA mismatch")
    prior=source.parent.strict_load(receipt_path.read_text("utf8"))
    if (prior.get("schema")!="hala.h418.owner-private-30k-fresh-original-drags.v1"
        or prior.get("selected_unique_original_train_images")!=ORIGINAL_IMAGE_N
        or prior.get("selected_bytes_sha256")!=H418_ROWS_SHA
        or prior.get("actual_all_30000_original_jpeg_bytes_sha_verified") is not True
        or prior.get("h417_dev144_image_sha_disjoint") is not True
        or prior.get("h417_train_image_sha_disjoint") is not True
        or prior.get("model_training_executed") is not False
        or prior.get("external_independent_ood") is not False):
        raise ValueError("H420 cannot reuse unverified source originals")
    proof_path=Path(source_task_receipt)
    if source.parent.sha_file(proof_path)!=H419_SOURCE_TASK_AUDIT_SHA:
        raise ValueError("H420 original 500K source-task isolation receipt changed")
    proof=source.parent.strict_load(proof_path.read_text("utf8"))
    if (proof.get("schema")!="hala.h419.private-original-source-task-isolation.v1" or
        proof.get("original_h60_source_actions_scanned")!=ORIGINAL_NATIVE_ACTIONS or
        proof.get("original_source_images_with_train_annotations")!=ORIGINAL_SOURCE_IMAGES or
        proof.get("train_new_original_screen_sha")!=ORIGINAL_IMAGE_N or
        proof.get("old_dev_original_screen_sha")!=96 or
        proof.get("fresh_dev_original_screen_sha")!=96 or
        any(value!={"image_sha_overlap":0,"original_source_task_sha_overlap":0}
            for value in proof.get("overlaps",{}).values()) or
        len(proof.get("overlaps",{}))!=3 or
        proof.get("independent_external_ood") is not False or
        proof.get("model_release_qualified") is not False):
        raise ValueError("H420 source-original tasks overlap frozen H419 DEV")
    selected={}
    for line in rows_path.open("r",encoding="utf8"):
        if not line.strip():continue
        rec=source.parent.strict_load(line)
        key=rec.get("image_sha256")
        if (rec.get("schema")!="hala.h418.original-native-drag-train.v1" or
            rec.get("split")!=SOURCE_SPLIT or
            rec.get("source_domain") not in FROZEN_COUNT_BY_DOMAIN or
            rec.get("source_zip") not in source.ALL_SHAS or
            rec.get("source_zip_sha256")!=source.ALL_SHAS[rec["source_zip"]] or
            not _hex_sha(key) or key in selected or
            not _hex_sha(rec.get("original_source_task_sha256")) or
            type(rec.get("image_width")) is not int or
            type(rec.get("image_height")) is not int or
            rec["image_width"]<=0 or rec["image_height"]<=0):
            raise ValueError("H420 original H418 source screen/label altered")
        native_tuple(rec["native_drag_completion"])
        selected[key]=rec
    if len(selected)!=ORIGINAL_IMAGE_N:
        raise ValueError("H420 expected exact 30K SHA-unique H418 TRAIN source screens")
    distribution=Counter(r["source_domain"] for r in selected.values())
    if distribution!={k:10000 for k in distribution} or len(distribution)!=3:
        raise ValueError("H420 original H418 three task families changed")
    return selected


def alternative(base,original_raw,zip_name):
    """A genuinely different pixel-target, not a differently named duplicate."""
    original_path=original_raw.get("image_path")
    if (original_raw.get("image_sha256")!=base["image_sha256"] or
        type(original_path) is not str or
        not original_path.startswith("images/"+base["source_domain"]+"/") or
        not original_path.endswith(".jpg") or
        ".." in original_path or
        original_raw.get("domain")!=base["source_domain"] or
        zip_name!=base["source_zip"]):
        raise ValueError("H420 different source image or ZIP falsely paired")
    # H60 original TRAIN annotation ZIPs contain pixel-identical JPEG
    # aliases under distinct immutable member paths (510 labelled cases
    # in the 30K source index). The SHA-pinned raw JPEG is the identity;
    # preserve EACH action's actual original member path, not the H419
    # first-action path. In the full run, BOTH paths must appear in the
    # original ZIP+labels index with the same source byte SHA, and the
    # chosen second path is re-opened and SHA-checked before admission.
    candidate=raw.original_record(original_raw,base["source_domain"],
                                  zip_name,base["image_sha256"])
    if (candidate["image_width"]!=base["image_width"] or
        candidate["image_height"]!=base["image_height"]):
        raise ValueError("H420 original source image dimensions mismatch")
    first=native_tuple(base["native_drag_completion"])
    second=native_tuple(candidate["native_drag_completion"])
    if first[:2]==second[:2]:
        return None
    width,height=base["image_width"],base["image_height"]
    shift=sum(abs(v-w) for before,after in zip(first[:2],second[:2])
                     for v,w in zip(before,after))
    ratio=shift/float(width+height)
    if ratio<=0:
        raise ValueError("H420 alternative had no genuinely different target")
    distinct_text=bool(base["intent"]!=candidate["intent"])
    id_sha=sha256((SEED+"|"+candidate["id"]).encode("utf8")).hexdigest()
    case={
       "schema":"hala.h420.original-second-native-drag-train.v1",
       "id":candidate["id"],
       "h419_first_action_id":base["id"],
       "original_source_task_sha256":candidate["original_source_task_sha256"],
       "source_domain":base["source_domain"],
       "source_zip":zip_name,
       "source_zip_sha256":source.ALL_SHAS[zip_name],
       "split":"train",
       "language":"en",
       "image_path":candidate["image_path"],
       "image_sha256":base["image_sha256"],
       "image_width":width,"image_height":height,
       "intent":candidate["intent"],
       "native_drag_completion":candidate["native_drag_completion"],
       "h419_first_native_action_sha256":sha256(
           source.canonical(base["native_drag_completion"])).hexdigest(),
       "additional_action_ratio_of_image_dims":round(ratio,6),
       "different_instruction_text":distinct_text,
       "same_verified_image_as_h419_train":True,
       "no_original_h417_or_h419_dev_image_or_task":True,
    }
    # Prioritize useful large native endpoint changes, prefer distinct text
    # only after geometry; stable deterministic tie break independent of GPU.
    rank=(ratio,int(distinct_text),id_sha)
    return rank,case


def select_alternatives(source_index,archives,baseline):
    best={}
    total_actions=0
    candidates=Counter()
    for z in archives:
        manifest=source.parent.strict_load(z.read("manifest.json").decode("utf8"))
        domain=manifest["source_domain"]
        filename=Path(z.filename).name
        with z.open("labels.jsonl") as stream:
            for line in stream:
                if not line.strip():continue
                total_actions+=1
                obj=source.parent.strict_load(line.decode("utf8"))
                image=obj["image_sha256"]
                original=baseline.get(image)
                if original is None:continue
                proof=source_index.get((domain,obj["image_path"]))
                if (proof is None or proof[1]!=filename or
                    proof[2]!=image):
                    raise ValueError("H420 original labelled raw JPEG not source-qualified")
                alt=alternative(original,obj,filename)
                if alt is None:continue
                candidates[domain]+=1
                if image not in best or alt[0]>best[image][0]:
                    best[image]=alt
    if total_actions!=ORIGINAL_NATIVE_ACTIONS:
        raise ValueError("H420 original 500,554 TRAIN annotated native actions altered")
    rows=[case for _,case in best.values()]
    rows.sort(key=lambda x:(x["source_domain"],sha256(
        (SEED+"|"+x["image_sha256"]).encode("utf8")).hexdigest()))
    by_domain=Counter(r["source_domain"] for r in rows)
    if (len(rows)!=FROZEN_CANDIDATE_N or
        by_domain!=FROZEN_COUNT_BY_DOMAIN or
        len({r["image_sha256"] for r in rows})!=len(rows) or
        len({r["id"] for r in rows})!=len(rows) or
        len(rows)<MIN_DISTINCT_ALTERNATIVES):
        raise ValueError("H420 original alternate actions diversity count drift")
    return rows,dict(candidates),dict(by_domain)


def verify_original_images(rows,index):
    from PIL import Image
    counts=Counter()
    for i,r in enumerate(rows):
        domain=r["source_domain"]
        z,name,expected=index[(domain,r["image_path"])]
        if name!=r["source_zip"] or expected!=r["image_sha256"]:
            raise ValueError("H420 original second task not from pinned source JPEG")
        blob=z.read(r["image_path"])
        if sha256(blob).hexdigest()!=expected or not blob.startswith(b"\xff\xd8\xff"):
            raise ValueError("H420 second action original TRAIN JPEG bytes changed")
        with Image.open(BytesIO(blob)) as im:
            if im.format!="JPEG" or im.size!=(r["image_width"],r["image_height"]):
                raise ValueError("H420 native new target/source ORIGINAL JPEG frame mismatch")
            im.verify()
        counts[domain]+=1
        if (i+1)%5000==0:
            print("H420_TRUE_ORIGINAL_SECOND_TARGET_SHA",i+1,"/",len(rows),
                  flush=True)
    return dict(counts)


def export(rows,candidate_counts,verified_domains,output_dir):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    out=output_dir/OUTPUT
    receipt=output_dir/RECEIPT
    sidecar=output_dir/"receipt.sha256"
    if out.exists() or receipt.exists() or sidecar.exists():
        raise FileExistsError("H420 new original action corpus CREATE ONLY")
    blob=b"".join(source.canonical(r)+b"\n" for r in rows)
    doc={
      "schema":"hala.h420.private-authentic-second-native-drags.v1",
      "status":"ORIGINAL_COMPLEMENTARY_SOURCE_BYTES_VERIFIED_MODEL_NOT_TRAINED",
      "source_original_nine_zip_SHA":source.ALL_SHAS,
      "original_first_30000_source_sha256":H418_ROWS_SHA,
      "original_first_30000_source_receipt_sha256":H418_RECEIPT_SHA,
      "original_task_heldout_isolation_receipt_sha256":H419_SOURCE_TASK_AUDIT_SHA,
      "total_original_actions_scanned":ORIGINAL_NATIVE_ACTIONS,
      "original_screens_in_baseline_h419_train":ORIGINAL_IMAGE_N,
      "new_original_distinct_complementary_native_actions":len(rows),
      "source_images_with_genuine_new_action":len(rows),
      "source_images_without_native_alternative":ORIGINAL_IMAGE_N-len(rows),
      "original_source_second_actions_jsonl_sha256":sha256(blob).hexdigest(),
      "original_source_second_actions_by_domain":verified_domains,
      "alternative_source_candidates_by_domain":candidate_counts,
      "new_actions_with_different_instruction_text":
          sum(r["different_instruction_text"] for r in rows),
      "new_actions_geometric_ratio_at_least_0_08":
          sum(r["additional_action_ratio_of_image_dims"]>=0.08 for r in rows),
      "all_selected_original_JPEG_bytes_SHA_verified":True,
      "all_actions_derived_from_original_train_only":True,
      "old_h417_and_new_h419_original_dev_tasks_quarantined":True,
      "all_30000_h419_original_screens_included_in_training":
          False,  # Some screens have no distinct second task.
      "source_only_firstparty_or_original_h60":True,
      "model_training_executed":False,
      "external_independent_ood":False,
      "model_release_qualified":False,
      "weights_public":False,
    }
    try:
        with out.open("xb") as f:f.write(blob)
        rec=json.dumps(doc,sort_keys=True,indent=2,allow_nan=False).encode()
        with receipt.open("xb") as f:f.write(rec)
        with sidecar.open("x",encoding="ascii",newline="\n") as f:
            f.write(sha256(rec).hexdigest())
    except BaseException:
        out.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return doc


def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--h415-zips",required=True)
    p.add_argument("--h418-zips",required=True)
    p.add_argument("--h418-30k",required=True)
    p.add_argument("--h418-receipt",required=True)
    p.add_argument("--h419-task-receipt",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    baseline=source_manifest(args.h418_30k,args.h418_receipt,args.h419_task_receipt)
    index,zips=source.open_archives(args.h415_zips,args.h418_zips)
    try:
        rows,candidates,_=select_alternatives(index,zips,baseline)
        verified=verify_original_images(rows,index)
        result=export(rows,candidates,verified,args.out)
        print("H420_TRUE_COMPLEMENTARY_NATIVE_SOURCE",
              json.dumps(result,sort_keys=True),flush=True)
    finally:
        for z in zips:z.close()

if __name__=="__main__":
    cli()

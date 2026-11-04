"""H418 owner-private curation of truly AVAILABLE original H61 native-drag source.

Original H61 rows are not proof of original media availability. This tool
admits only ZIP SHA-pinned TRAIN images, recomputes every selected JPEG
byte hash, excludes H417 image-level development hashes, deduplicates by
real SHA256 pixels, and emits a create-only ORIGINAL training manifest.

No model fitting, weight export, medical records or release sign-off.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import zipfile

from h417.kaggle import h417_native_drag_lora as parent

FROZEN_SHA="038baec02070a48a941f320d336f009eaa640967906ca64031b52aa768c8ac2d"
FROZEN_ROWS=64300
SOURCE_REV=parent.SOURCE_REV
EXTRA_SHAS={
  "h60-dragon-train-slide_resize-0001.hala.bin":
    "6e5ad052a460fe23eb865a15fbf83694d248970c3eb158118bb83feff4c696db",
  "h60-dragon-train-text_highlight-0001.hala.bin":
    "c07191900e1a4f17f65795381a57b12d91071d4dc580b87a6e4875293f5160af",
  "h60-dragon-train-text_highlight-0002.hala.bin":
    "ae07e67f085d1749f19747af25fc3571768cc28bfda1e4ebf980ec0f207a3005",
  "h60-dragon-train-text_highlight-0003.hala.bin":
    "4170b4687e462c045ecbfb7198324110a3db54d5b2b774b3fdb6bdbce3586b74",
  "h60-dragon-train-text_highlight-0004.hala.bin":
    "c4824f510cb9d27bec60bed536aba81fd3fdfb50ee827c9e2a121f730a16d782",
}
BASE_SHAS={name:sha for name,sha in parent.SOURCE_SHAS.values()}
ALL_SHAS={**BASE_SHAS,**EXTRA_SHAS}
EXPECTED_AVAILABLE_SHA=30491
EXPECTED_OLD_BASE_ORIGINAL=13165
EXPECTED_NEW_EXTENSION=17326
OUT_ROWS="h418_original_train_verified.jsonl"
OUT_MANIFEST="h418_original_source_manifest.json"


def canonical(record):
    return json.dumps(record,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode("utf8")


def _hex(value):
    return (type(value) is str and len(value)==64
            and all(c in "0123456789abcdef" for c in value))


def frozen_source(rows_path, manifest_path):
    rows_path=Path(rows_path);manifest_path=Path(manifest_path)
    if rows_path.name!="h61_drag_curriculum.jsonl":
        raise ValueError("H418 original frozen H61 source name required")
    if parent.sha_file(rows_path)!=FROZEN_SHA:
        raise ValueError("H418 original 64.3K frozen TRAIN bytes SHA changed")
    man=parent.strict_load(manifest_path.read_text("utf8"))
    if (man.get("schema")!="hala.training.manifest.v1"
        or man.get("dataset_sha256")!=FROZEN_SHA
        or man.get("train_rows")!=FROZEN_ROWS
        or man.get("split")!="train"
        or man.get("source_kind")!="rights_cleared_public"
        or man.get("source_dataset")!="Hcompany/DragOn"
        or man.get("source_revision")!=SOURCE_REV
        or man.get("source_license")!="Apache-2.0"
        or man.get("no_benchmark_eval_source") is not True
        or man.get("training_executed") is not False):
        raise ValueError("H418 source history not original TRAIN-only")
    media=man.get("media")
    if type(media) is not list:
        raise ValueError("H418 frozen source media manifest missing")
    allowed={}
    for m in media:
        if (type(m) is not dict or
            type(m.get("path")) is not str or
            not _hex(m.get("sha256")) or
            m["path"] in allowed):
            raise ValueError("H418 original H61 media provenance repeated")
        allowed[m["path"]]=m["sha256"]
    rows=[parent.strict_load(line.decode("utf8")) for line
          in rows_path.read_bytes().splitlines() if line.strip()]
    if len(rows)!=FROZEN_ROWS:
        raise ValueError("H418 64300 actual frozen source rows missing")
    for r in rows:
        if (type(r) is not dict or r.get("split")!="train"
            or not _hex(r.get("image_sha256"))
            or allowed.get(r.get("image_path"))!=r["image_sha256"]):
            raise ValueError("H418 annotated original TRAIN media tampered")
    return rows,man


def open_archives(base_root,extension_root):
    base_root=Path(base_root);extension_root=Path(extension_root)
    opened=[]
    source={}
    try:
        for name,digest in ALL_SHAS.items():
            path=(extension_root if name in EXTRA_SHAS else base_root)/name
            if path.name!=name or not path.is_file():
                raise ValueError("H418 original private archive missing: "+name)
            if parent.sha_file(path)!=digest:
                raise ValueError("H418 private ZIP byte SHA mismatch: "+name)
            z=zipfile.ZipFile(path)
            opened.append(z)
            m=parent.strict_load(z.read("manifest.json").decode("utf8"))
            parts=name[len("h60-dragon-train-"):-len(".hala.bin")]
            domain=parts.rsplit("-",1)[0]
            if (m.get("schema")!="hala.h60.dragon-train-shard.v1"
                or m.get("source_commit_sha")!=SOURCE_REV
                or m.get("source_domain")!=domain
                or m.get("split")!="train"
                or m.get("official_eval_or_test_used") is not False
                or m.get("benchmark_answer_leakage") is not False
                or m.get("source_license")!="Apache-2.0"):
                raise ValueError("H418 original H60 TRAIN manifest does not qualify")
            annotations=z.read("labels.jsonl")
            if sha256(annotations).hexdigest()!=m.get("labels_sha256"):
                raise ValueError("H418 owner-original H60 annotations modified")
            for member in z.namelist():
                if not member.startswith("images/") or not member.endswith(".jpg"):
                    continue
                key=(domain,member)
                if key in source:
                    raise ValueError("H418 duplicate JPEG path across private shards")
                source[key]=(z,name)
        return source,opened
    except BaseException:
        for z in opened:z.close()
        raise


def h417_dev_source_sha(h417_root):
    # Re-derive exact held-back original H417 source images rather than
    # trusting a hand-edited list of development SHA values.
    h417_root=Path(h417_root)
    originals=parent.only_input_file
    archives=None
    try:
        parent.only_input_file=lambda name:h417_root/name
        archives,raw,curriculum,bilingual=parent.get_research_sources()
        _,dev,_=parent.partition(raw,curriculum,bilingual)
        if len(dev)!=144 or len({r["image_sha256"] for r in dev})!=144:
            raise ValueError("H418 source H417 development source count changed")
        return {r["image_sha256"] for r in dev}
    finally:
        parent.only_input_file=originals
        if archives:
            for z in archives.values():z.close()


def select(rows,source,heldout,*,verify_bytes=True,
           expected_available=EXPECTED_AVAILABLE_SHA,min_selected=30000):
    if type(heldout) is not set or len(heldout)!=144 or not all(
        _hex(x) for x in heldout):
        raise ValueError("H418 original H417 frozen development SHA quarantine missing")
    found={}
    eligible=Counter()
    excluded=Counter()
    selected=[]
    for row in rows:
        path=row["image_path"]
        parts=path.split("/",1)
        if len(parts)!=2 or parts[0] not in parent.DOMAINS:
            excluded["non_original_domain_or_firstparty"]+=1
            continue
        domain,member=parts
        actual=source.get((domain,member))
        if actual is None:
            excluded["original_image_archive_unavailable"]+=1
            continue
        eligible[domain]+=1
        if row["image_sha256"] in heldout:
            excluded["h417_dev_sha_quarantined"]+=1
            continue
        if row["image_sha256"] in found:
            excluded["duplicate_source_pixel_sha"]+=1
            continue
        z,name=actual
        if verify_bytes:
            raw=z.read(member)
            if sha256(raw).hexdigest()!=row["image_sha256"]:
                raise ValueError("H418 original JPEG bytes differ from frozen H61 SHA")
            if not raw.startswith(b"\xff\xd8\xff"):
                raise ValueError("H418 original image not JPEG source")
        # Store canonical member path for existing native drag converter.
        converted=dict(row)
        converted["original_h61_image_path"]=path
        converted["image_path"]=member
        converted["source_domain"]=domain
        converted["source_zip"]=name
        converted["source_zip_sha256"]=ALL_SHAS[name]
        selected.append(converted)
        found[row["image_sha256"]]=True
    if sum(eligible.values())!=expected_available:
        raise ValueError("H418 source coverage drift from owner archive inventory")
    if (sum(eligible[d] for d in ("sheet","slide_resize","text_highlight"))
        !=expected_available or eligible.get("slider",0)!=0):
        raise ValueError("H418 unexpected original H61 source domains")
    if len(selected)<min_selected or len(selected)>expected_available:
        raise ValueError("H418 insufficient original source pixel-distinct training")
    if len(found)!=len(selected):
        raise ValueError("H418 duplicate first-party/source image SHA")
    if {r["image_sha256"] for r in selected}&heldout:
        raise ValueError("H418 leaked H417 source-dev image into TRAIN")
    return selected,dict(eligible),dict(excluded)


def export(rows,eligible,excluded,out_path,*,zip_verified=True):
    out_path=Path(out_path)
    out_path.mkdir(parents=True,exist_ok=True)
    target=out_path/OUT_ROWS
    meta=out_path/OUT_MANIFEST
    if target.exists() or meta.exists():
        raise FileExistsError("H418 source export is CREATE ONLY")
    if not zip_verified:
        raise ValueError("H418 cannot export without exact original JPEG SHA replay")
    blob=b"".join(canonical(r)+b"\n" for r in rows)
    rec={
      "schema":"hala.h418.owner-private-30k-source-selection.v1",
      "status":"COMPLETE_SOURCE_ONLY_NO_MODEL_TRAINING",
      "source":"Hcompany/DragOn",
      "source_revision":SOURCE_REV,
      "source_split":"train",
      "source_frozen_h61_sha256":FROZEN_SHA,
      "all_archive_sha256_verified":True,
      "all_selected_JPEG_byte_sha256_verified":True,
      "training_performed":False,
      "independent_external_ood":False,
      "official_real_GUI_train_rights_independently_approved":False,
      "weights_public":False,
      "model_release_qualified":False,
      "h417_dev_sha_excluded":True,
      "selected_train_source_unique_pixels":len(rows),
      "selected_train_rows":len(rows),
      "selected_sha256":sha256(blob).hexdigest(),
      "source_available_by_domain":eligible,
      "source_excluded":excluded,
      "archive_sha256":ALL_SHAS,
    }
    try:
        with target.open("xb") as file:file.write(blob)
        with meta.open("xb") as file:file.write(
            json.dumps(rec,sort_keys=True,indent=2,allow_nan=False).encode())
    except BaseException:
        target.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        raise
    return rec


def run_cli():
    p=argparse.ArgumentParser()
    p.add_argument("--h61",required=True)
    p.add_argument("--h61-manifest",required=True)
    p.add_argument("--h415-zips",required=True)
    p.add_argument("--h418-zips",required=True)
    p.add_argument("--h417-originals",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    rows,_=frozen_source(args.h61,args.h61_manifest)
    holdout=h417_dev_source_sha(args.h417_originals)
    index,archives=open_archives(args.h415_zips,args.h418_zips)
    try:
        selected,eligible,excluded=select(rows,index,holdout,verify_bytes=True)
        result=export(selected,eligible,excluded,args.out)
    finally:
        for z in archives:z.close()
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    run_cli()

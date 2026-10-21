"""Audit every staged original TAR range; compare reassembled stream with HF LFS SHA.

This verifies source archive bytes, NOT source JSON/JPEG semantics, Kaggle
PRIVATE storage or that model training was executed. No raw data uploaded.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from h59.master_data_plan import DOMAINS,check_plan

def verify_archive(domain,planned_windows,folder,read_chunk=4*1024*1024):
    if domain not in DOMAINS or not isinstance(planned_windows,list):
        raise ValueError("Official source domain required")
    expected=DOMAINS[domain]
    windows=[v for v in planned_windows if v["source_domain"]==domain]
    if not windows:raise ValueError("No window coverage")
    folder=Path(folder)
    whole=hashlib.sha256()
    cursor=0;receipts=[]
    for v in windows:
        if (v["byte_start"]!=cursor or
            v["upstream_full_tar_lfs_sha256"]!=expected["upstream_lfs_sha256"]):
            raise ValueError("Gap/overlap or source SHA mutation")
        ident=v["id"]
        filename=folder/(ident+".tarwindow")
        rp=folder/(ident+".receipt.json")
        if not filename.exists() or not rp.exists():
            raise FileNotFoundError("One or more original source windows NOT MATERIALIZED: "+ident)
        r=json.loads(rp.read_text("utf8"))
        if (r.get("schema")!="hala.h59.individual-train-tar-window-receipt.v1" or
            r.get("original_window_id")!=ident or
            r.get("source_domain")!=domain or
            r.get("original_commit_sha")!="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5" or
            r.get("bytes_verified")!=v["expected_length"] or
            r.get("range_start")!=v["byte_start"] or
            r.get("range_end_inclusive")!=v["byte_end_inclusive"] or
            not re.fullmatch("[a-f0-9]{64}",str(r.get("original_window_sha256")))):
            raise ValueError("Source receipt does not match pinned original member")
        if filename.stat().st_size!=v["expected_length"]:
            raise ValueError("Original TAR byte window incomplete")
        per=hashlib.sha256()
        with filename.open("rb") as source:
            for piece in iter(lambda:source.read(read_chunk),b""):
                per.update(piece);whole.update(piece)
        if per.hexdigest()!=r["original_window_sha256"]:
            raise ValueError("Original byte window altered after retrieval")
        receipts.append(ident)
        cursor+=v["expected_length"]
    if cursor!=expected["bytes"]:
        raise ValueError("Full original TAR still incomplete")
    if whole.hexdigest()!=expected["upstream_lfs_sha256"]:
        raise ValueError("Entire TRAIN TAR failed official pinned Hugging Face LFS SHA256")
    return dict(domain=domain,original_tar_sha256=whole.hexdigest(),
      original_source_bytes_verified=cursor,source_windows_qualified=len(receipts),
      task_records_parsed=False,original_image_task_pairing_complete=False,
      provider_private_kaggle_upload_verified=False,gpu_training_performed=False)

def verify_all(plan_path,folder):
    plan=json.loads(Path(plan_path).read_text("utf8"))
    check_plan(plan)
    records=[verify_archive(domain,plan["windows"],folder) for domain in DOMAINS]
    if sum(x["original_source_bytes_verified"] for x in records)!=plan["original_tar_bytes"]:
        raise ValueError("Original four-source TAR coverage incomplete")
    return dict(source_original_tars_full_SHA_verified=True,domains=records,
      source_paired_tasks_indexed=False,millions_of_training_rows_ready=False,
      gpu_training_executed=False)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--plan",default="h59/full_source_windows.json")
    p.add_argument("--windows-folder",required=True)
    args=p.parse_args()
    print(json.dumps(verify_all(args.plan,args.windows_folder),sort_keys=True))

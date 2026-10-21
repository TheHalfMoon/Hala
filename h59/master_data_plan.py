"""H59 reproducible full original DragOn TRAIN byte-range inventory (metadata ONLY).

A planned HTTP byte window is not downloaded data, paired annotations, model
training or a Kaggle dataset. All windows must be independently read,
verified, materialized in a provider-permitted PRIVATE storage location, and
indexed before marking source preparation complete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

REV="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
REPO="Hcompany/DragOn"
SCHEMA="hala.h59.full-million-drag-train-byte-range-plan.v1"
WINDOW=512*1024*1024
DOMAINS={
    "text_highlight":{"bytes":29874024960,"images":100000,"tasks":1000000,"ordered":"false-prevalent","upstream_lfs_sha256":"130d1a47add164bb84849abc3ae5ea29d4dd58a892082c887bef9866986515f8"},
    "sheet":{"bytes":14258661376,"images":38332,"tasks":996632,"ordered":"mixed","upstream_lfs_sha256":"19a78475e7bc4d66a3f1f20b49ce12b6760a820270365f5ef30f3fd447d4cded"},
    "slide_resize":{"bytes":9915637248,"images":82680,"tasks":992160,"ordered":"true-prevalent","upstream_lfs_sha256":"16414ddb6e4b7c7f07e93e1548468b21dd29ab5e028fb0931e6b973bcca0b89f"},
    "slider":{"bytes":7146830848,"images":65000,"tasks":571350,"ordered":"true-prevalent","upstream_lfs_sha256":"2455d5b4a164d4ff3d6cf41d92d24ad2e27db29e28ea16f5585b55956356a33e"}
}
KNOWN_STORED={
 "dataset1":"abdulazizshehri/hala-gui-training-library-private",
 "dataset2":"abdulazizshehri/hala-dragon-multidomain-train-private",
 "original_hcompany_train_screenshots":204,
 "original_hcompany_train_drag_annotations":2346
}

def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf8")

def build():
    windows=[]
    for domain,info in DOMAINS.items():
        for index,start in enumerate(range(0,info["bytes"],WINDOW)):
            length=min(WINDOW,info["bytes"]-start)
            if start%512 or length%512 or length<512:
                raise ValueError("Each source TAR window must be 512-byte aligned")
            windows.append({
                "id":f"h59-{domain}-{index:04d}",
                "source_domain":domain,"url":f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/train/{domain}.tar?download=true",
                "byte_start":start,"byte_end_inclusive":start+length-1,
                "expected_length":length,"total_tar_bytes":info["bytes"],"upstream_full_tar_lfs_sha256":info["upstream_lfs_sha256"],
                "expected_content_range":f"bytes {start}-{start+length-1}/{info['bytes']}",
                "source_window_sha256":None,
                "window_downloaded":False,"provider_private_storage_verified":False,
                "tar_member_index_complete":False,
                "verified_original_images":0,"verified_original_tasks":0,
                "reusable_train_examples_ready":0,
                "official_eval_or_test_used":False
            })
    summary=dict(schema=SCHEMA,source_repo=REPO,source_revision=REV,
       split="train",source_license="Apache-2.0",
       original_tar_bytes=sum(v["bytes"] for v in DOMAINS.values()),
       advertised_original_train_images=sum(v["images"] for v in DOMAINS.values()),
       advertised_original_train_tasks=sum(v["tasks"] for v in DOMAINS.values()),
       original_upstream_counts_not_yet_locally_ingested=True,
       source_domains={k:{**v,"split":"train","upstream_full_tar_verified":False,
                           "all_original_media_downloaded":False}
                       for k,v in DOMAINS.items()},
       existing_source_qualified_private_kaggle=KNOWN_STORED,
       planned_window_bytes=WINDOW,planned_windows=len(windows),
       downloaded_windows_count=0,source_file_windows_readback_sha_verified_count=0,
       fully_indexed_unique_original_images=0,
       fully_materialized_original_drag_tasks=0,
       full_private_kaggle_uploaded=False,
       original_tar_sha256_all_verified=False,train_gpu_steps=0,
       model_checkpoint_trained=False,independent_model_benchmark_done=False,
       no_eval_test_or_benchmark_answer_leakage=True,
       windows=windows)
    return summary

def check_plan(x):
    if (not isinstance(x,dict) or x.get("schema")!=SCHEMA or
        x.get("source_revision")!=REV or x.get("source_repo")!=REPO or
        x.get("split")!="train" or
        x.get("advertised_original_train_images")!=286012 or
        x.get("advertised_original_train_tasks")!=3560142 or
        x.get("planned_window_bytes")!=WINDOW or
        x.get("full_private_kaggle_uploaded") is not False or
        x.get("train_gpu_steps")!=0 or
        x.get("model_checkpoint_trained") is not False or
        x.get("fully_indexed_unique_original_images")!=0 or
        x.get("fully_materialized_original_drag_tasks")!=0):
        raise ValueError("H59 planned original TRAIN catalog must not claim downloaded training")
    expected=build()
    if x!=expected:
        raise ValueError("Missing, modified or source-overlapping planned original train byte windows")
    for dom,v in DOMAINS.items():
        ranges=[p for p in x["windows"] if p["source_domain"]==dom]
        if sum(p["expected_length"] for p in ranges)!=v["bytes"]:
            raise ValueError("Source TAR coverage not exact")
    return dict(upstream_images=286012,upstream_tasks=3560142,
      planned_windows=len(x["windows"]),planned_bytes=x["original_tar_bytes"],
      source_train_materialized=False,neural_training_executed=False)

def save(path):
    path=Path(path)
    if path.exists():raise FileExistsError("Create-only fixed original full-source manifest")
    blob=canonical(build())
    with path.open("xb") as file:file.write(blob)
    return dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
         **check_plan(json.loads(blob)))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["build","verify"])
    p.add_argument("--file",default="h59/full_source_windows.json")
    a=p.parse_args()
    print(json.dumps(save(a.file) if a.command=="build" else
         check_plan(json.loads(Path(a.file).read_text("utf8"))),sort_keys=True))

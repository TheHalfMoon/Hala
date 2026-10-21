"""H59 resumable, bounded retrieval of ONE immutable original TRAIN TAR byte window.

Run on authorized PRIVATE data processing compute only. Not a GPU trainer,
not a Kaggle session extender and not a publisher. HTTP Range MUST be 206 with
the expected Content-Range. Does not silently mark remote Kaggle storage done.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import urllib.request
from h59.master_data_plan import check_plan

def get_window(plan_path,window_id,directory,*,allow_network=False,block_bytes=1024*1024):
    plan=json.loads(Path(plan_path).read_text("utf8"))
    check_plan(plan)
    windows={p["id"]:p for p in plan["windows"]}
    if window_id not in windows or len(windows)!=len(plan["windows"]):
        raise ValueError("A single exact planned train window ID is required")
    if not allow_network:
        raise ValueError("Explicit --download train source opt-in required; dry run never fetches")
    if block_bytes>4*1024*1024 or block_bytes<4096:
        raise ValueError("Bounded streaming network block required")
    entry=windows[window_id]
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    part=directory/(window_id+".part")
    target=directory/(window_id+".tarwindow")
    receipt=directory/(window_id+".receipt.json")
    if target.exists() or receipt.exists():
        raise FileExistsError("Existing original source or receipt requires independent verification")
    existing=part.stat().st_size if part.exists() else 0
    if existing>=entry["expected_length"]:
        if existing!=entry["expected_length"]:
            raise ValueError("Source partial window larger than target")
    else:
        start=entry["byte_start"]+existing
        end=entry["byte_end_inclusive"]
        expected_range=f"bytes {start}-{end}/{entry['total_tar_bytes']}"
        request=urllib.request.Request(entry["url"],headers={
            "Range":f"bytes={start}-{end}",
            "User-Agent":"Hala-H59-Source-Only-Bounded-TRAIN-Download"})
        with urllib.request.urlopen(request,timeout=120) as response:
            if (response.status!=206 or
                response.headers.get("Content-Range")!=expected_range):
                raise ValueError("Upstream TRAIN server did not honor exact pinned HTTP 206 byte range")
            # Range Content-Length may be omitted for chunked responses; when present verify.
            size=response.headers.get("Content-Length")
            if size is not None and int(size)!=(end-start+1):
                raise ValueError("Upstream Content-Length mismatch")
            remaining=end-start+1
            with part.open("ab" if existing else "xb") as file:
                while remaining:
                    chunk=response.read(min(block_bytes,remaining))
                    if not chunk:raise IOError("Source tar window truncated, preserve .part for resume")
                    file.write(chunk)
                    remaining-=len(chunk)
                file.flush();os.fsync(file.fileno())
    if part.stat().st_size!=entry["expected_length"]:
        raise ValueError("Window not yet entirely downloaded")
    digest=hashlib.sha256()
    with part.open("rb") as reader:
        for chunk in iter(lambda:reader.read(block_bytes),b""):
            digest.update(chunk)
    hexdigest=digest.hexdigest()
    os.replace(part,target)
    detail=dict(schema="hala.h59.individual-train-tar-window-receipt.v1",
        original_window_id=window_id,source_repo=plan["source_repo"],
        original_commit_sha=plan["source_revision"],source_domain=entry["source_domain"],
        split="train",range_start=entry["byte_start"],
        range_end_inclusive=entry["byte_end_inclusive"],
        bytes_verified=entry["expected_length"],original_window_sha256=hexdigest,
        saved_original_window_path=str(target),
        provider_private_kaggle_uploaded=False,
        provider_private_kaggle_readback_sha_verified=False,
        train_tasks_parsed=0,train_images_verified=0,
        gpu_training_steps=0)
    receipt.write_text(json.dumps(detail,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return detail

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--plan",default="h59/full_source_windows.json")
    p.add_argument("--window-id",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--download",action="store_true",
                   help="Explicit permitted source-download opt-in. Defaults to offline rejection.")
    args=p.parse_args()
    print(json.dumps(get_window(args.plan,args.window_id,args.output_dir,
                                allow_network=args.download),sort_keys=True))

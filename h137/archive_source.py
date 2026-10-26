"""H137 private screenshot-only archive for Kaggle; vendor JS stays owner local."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED

from h137.source import check,digest

def pack(source_dir,destination):
    source_dir,destination=map(Path,(source_dir,destination))
    if destination.exists():raise ValueError("H137 evaluation archive create-only")
    src=check(source_dir)
    expected=["manifest.json"]+[c["image"] for c in src["cases"]]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,"x",compression=ZIP_STORED) as z:
        for name in expected:z.write(source_dir/name,arcname=name)
    with ZipFile(destination,"r") as z:
        if z.namelist()!=expected:raise ValueError("H137 private only manifest and exact 36 images")
        for case in src["cases"]:
            if digest(z.read(case["image"]))!=case["image_sha256"]:
                raise ValueError("H137 compressed screenshot SHA mismatch")
    return destination

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    f=pack(a.source,a.output)
    print("H137 EVAL-ONLY OWNER-PRIVATE SOURCE PACK VERIFIED",f.stat().st_size)

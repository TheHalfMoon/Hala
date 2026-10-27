"""Owner-private Kaggle H142 TRAIN64+DEV16+prior 2B LoRA upload pack.

No H140 OOD BLIND, no consumed H141 model-evaluated original screenshot,
no unlicensed third-party code. Kaggle PRIVATE default, owner only.
"""
import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from h140.qualification import digest
from h142.qualify import verify,FOLDERS

DATASET_ID="abdulazizshehri/hala-h142-private-visible-heading-sft-v1"
BASE_ADAPTER_ZIP="h140-private-original-v2-rank8-lora.zip"
BASE_ADAPTER_SHA="568550c7bf2165821e5898339174e2dfbccd40c117e9a6cda6c3d9879a1042c0"

def pack(owner_root):
    root=Path(owner_root)
    qualified=verify(root)
    destination=root/"private-h142"/"kaggle-private-training-v1"
    if destination.exists():
        raise ValueError("H142 owner-private dataset is create-only")
    destination.mkdir(parents=True)
    source=root/"private-h142"
    output={}
    for split in ("train","dev"):
        origin=source/FOLDERS[split]
        archive=destination/(FOLDERS[split]+".zip")
        rows=json.loads((origin/"manifest.json").read_text("utf8"))["cases"]
        with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=2) as z:
            z.write(origin/"manifest.json",arcname="manifest.json")
            for row in rows:
                png=origin/row["image"]
                if digest(png.read_bytes())!=row["image_sha256"]:
                    raise ValueError("H142 original native screenshot changed in ZIP")
                z.write(png,arcname=row["image"])
        with ZipFile(archive) as z:
            if (z.testzip() is not None or len(z.namelist())!=len(rows)+1 or
                digest(z.read("manifest.json"))!=
                    qualified[split]["manifest_sha256"]):
                raise ValueError("H142 owner source archive not fully qualified")
        output[split]={"cases":len(rows),"zip_sha":digest(archive.read_bytes()),
                       "zip_bytes":archive.stat().st_size}
    adapter=(root/"private-h141"/"kaggle-private-visible-probe-v1"/
             BASE_ADAPTER_ZIP)
    if digest(adapter.read_bytes())!=BASE_ADAPTER_SHA:
        raise ValueError("H142 old V2 2B LoRA SHA/owner ZIP drift")
    shutil.copy2(adapter,destination/BASE_ADAPTER_ZIP)
    metadata={
        "id":DATASET_ID,
        "title":"hala-h142-private-visible-heading-sft-v1",
        "licenses":[{"name":"other"}],
    }
    (destination/"dataset-metadata.json").write_text(
        json.dumps(metadata,sort_keys=True,indent=2)+"\n","utf8")
    return {"status":"H142_ONLY_PRIVATE_TRAIN_DEV_AND_PRIOR_ADAPTER_PACKED",
            "qualified_original_sources":qualified,
            "files":output,
            "model_2b_adapter_sha":BASE_ADAPTER_SHA,
            "blind_training_inputs":0,"paid_compute":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    args=p.parse_args()
    print(json.dumps(pack(args.owner_root),sort_keys=True,indent=2))

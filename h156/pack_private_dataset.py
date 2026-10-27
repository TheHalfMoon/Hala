"""Owner-private H156 Kaggle dataset packer; never publish or overwrite."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from h156.audit_train_dev import audit

DATASET_ID="abdulazizshehri/hala-h156-private-typed-action-pilot-v1"
ADAPTER_SHA="58ec4f6e026b5fc5c8be9a74433d3ca365f1c93680d84b2d148a86b8725e2b8c"


def pack(owner):
    owner=Path(owner)
    check=audit(owner)
    source=owner/"private-h156"/"miniwob-action-train-dev-v2"
    root=owner/"private-h156"/"kaggle-private-typed-action-v1"
    if root.exists():
        raise FileExistsError("H156 private Kaggle dataset create-only")
    weights=owner/"private-h153"/"kaggle-private-eval-v1"/"h142-private-visible-column-lora.zip"
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=ADAPTER_SHA:
        raise ValueError("H156 cannot alter frozen H142 private research adapter")
    root.mkdir(parents=True,exist_ok=False)
    archive=root/"h156-source-images.zip"
    with ZipFile(archive,"x",compression=ZIP_DEFLATED,compresslevel=3) as z:
        z.write(source/"manifest.json","manifest.json")
        manifest=json.loads((source/"manifest.json").read_text("utf8"))
        for row in manifest["private_cases"]:
            name=row["image"]
            image=source/"images"/name
            if hashlib.sha256(image.read_bytes()).hexdigest()!=row["image_sha256"]:
                raise ValueError("Original MiniWoB source media changed")
            z.write(image,"images/"+name)
    with ZipFile(archive) as z:
        if (z.testzip() is not None or len(z.namelist())!=61 or
            hashlib.sha256(z.read("manifest.json")).hexdigest()!=check["source_sha256"]):
            raise ValueError("H156 private Kaggle archive integrity failed")
    shutil.copyfile(weights,root/"h142-private-visible-column-lora.zip")
    metadata={
        "id":DATASET_ID,"title":"hala-h156-private-typed-action-pilot-v1",
        "licenses":[{"name":"other"}],
    }
    (root/"dataset-metadata.json").write_text(
        json.dumps(metadata,indent=2)+"\n",encoding="utf8")
    return {"status":"H156_PRIVATE_DATASET_PACKAGE_COMPLETE",
            "dataset_id":DATASET_ID,
            "source_manifest_sha256":check["source_sha256"],
            "source_zip_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "model_zip_sha256":ADAPTER_SHA,
            "source_images":60,"train":48,"dev":12,
            "private_only":True,"h140_blind_calls":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    o=p.parse_args()
    print(json.dumps(pack(o.owner_root),indent=2,sort_keys=True))

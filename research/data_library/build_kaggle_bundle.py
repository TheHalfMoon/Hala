"""Build a private Kaggle upload from original H50 TRAIN only; no network call."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
from h50.counterfactual import audit

ROOT=Path(__file__).resolve().parents[2]
DATASET="abdulazizshehri/hala-gui-training-library"
RECEIPTS=("authoring_sft.jsonl","sft_manifest.json",
          "source_provenance.json","click_replay.json")
def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def build(folder):
    folder=Path(folder)
    if folder.exists():raise FileExistsError("Kaggle staging is create-only")
    source=ROOT/"h50/assets"
    rows,m=audit(source)
    if len(rows)!=384 or m["source_screenshots"]!=64 or m["dev_used_as_training"]!=0:
        raise ValueError("H50 source admission failed")
    names=sorted([*RECEIPTS,*(f"png/{Path(x['path']).name}" for x in m["media"])])
    if len(names)!=68 or len(set(names))!=68:raise ValueError("Incorrect original TRAIN inventory")
    originals={}
    for name in names:
        if any(s in name.lower() for s in ("evaluator","dev_public","test")):
            raise ValueError("Reserved evaluation material cannot be uploaded")
        originals[name]=(source/name).read_bytes()
    folder.mkdir(parents=True)
    archive=folder/"hala-h50-train-original.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:
            info=zipfile.ZipInfo("h50-train/"+name,(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,originals[name],compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist())!={"h50-train/"+name for name in names}:
            raise ValueError("Archive CRC or inventory failed")
    meta=dict(schema="hala.kaggle.train-library.v1",private=True,
        train_only=True,model_training_executed=False,
        external_competitor_data_included=False,evaluation_material_included=False,
        unique_screenshots=64,supervision_rows=384,
        languages={"ar":192,"en":192},
        h50_dataset_sha256=m["dataset_sha256"],
        archive_sha256=hash_bytes(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,
        files={name:hash_bytes(originals[name]) for name in names},
        origin="TheHalfMoon/Hala h50/assets",
        url="https://www.kaggle.com/datasets/"+DATASET)
    (folder/"library-manifest.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (folder/"README.md").write_text(
        "# Hala GUI Training Library (PRIVATE)\n\n"
        "Original first-party H50 synthetic TRAIN data: 64 unique screenshots, "
        "384 counterfactual target annotations, 192 Arabic + 192 English. "
        "No third-party dataset bytes, private user data, government accounts, "
        "clinical records, H49 DEV, benchmark TEST, Holo4 evaluation trajectories "
        "or neural model outputs. These are supervised labels, not evidence of model accuracy.\n"
        "See library-manifest.json for file-level SHA256 hashes. Track acquisition "
        "and actual training for future sources in Hala's GitHub data-library checklist.\n",
        encoding="utf-8")
    (folder/"dataset-metadata.json").write_text(json.dumps({
        "title":"Hala GUI Training Library",
        "subtitle":"Private synthetic bilingual GUI training examples",
        "description":"Only first-party synthetic H50 train images and targets; no evaluation/private competitor images.",
        "id":DATASET,"licenses":[{"name":"CC0-1.0"}]
    },indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(folder),"sha256":meta["archive_sha256"],
                      "bytes":meta["archive_bytes"],"images":64,"labels":384,
                      "kaggle_private_by_default":True}))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    build(p.parse_args().output)

"""Create-only complete private Kaggle *version 2* of Hala first-party library.

Preserves original H50 v1 archive exactly and adds source-bound H52 train.
No Kaggle network call; caller explicitly runs authenticated private version.
"""
from __future__ import annotations
import hashlib,json,shutil,zipfile
from pathlib import Path
from h52.audit import audit
from research.data_library.build_kaggle_bundle import ROOT as ROOT_H50,build as build_h50

ROOT=Path(__file__).resolve().parents[3]
V1_UPLOAD_SHA="783426a4f8ae10c2ce85fa6b9cc861f69857e64372e1a4dc8a4cf81b8053c37c"
V1_LABEL_SHA="cb1a8529eaf0f82272d75a459c1634766ee5ec7694f6b21affa2045b11d02dd6"
H52_LABEL_SHA="bca090f0a31aee88fba0cf47aaadaa895ca1fc7ede7c41fbf123d65d6ccb4c73"

def sha(b):return hashlib.sha256(b).hexdigest()

def build(folder):
    folder=Path(folder)
    if folder.exists():raise FileExistsError("Never overwrite existing Kaggle stage")
    result=audit(ROOT/"h52/assets")
    if result["dataset_sha256"]!=H52_LABEL_SHA:raise ValueError("H52 actual browser replay not exact frozen material")
    v1=folder.parent/(folder.name+"-v1")
    if v1.exists():raise FileExistsError("v1 temp already exists")
    build_h50(v1)
    h50zip=v1/"hala-h50-train-original.zip"
    if sha(h50zip.read_bytes())!=V1_UPLOAD_SHA:
        raise ValueError("Cannot preserve original Kaggle v1 archive bit for bit")
    folder.mkdir(parents=True)
    original=ROOT/"h52/assets"
    names=["authoring_sft.jsonl","sft_manifest.json","capture.json",
           "plans.json","source.json"]+["html/"+p.name for p in sorted((original/"html").glob("*.html"))]+[
           "png/"+p.name for p in sorted((original/"png").glob("*.png"))]
    if len(names)!=389 or len(set(names))!=389:raise ValueError("H52 source/PNG count mismatch")
    archive=folder/"hala-h52-multistep-train.zip"
    digestlist={}
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(names):
            b=(original/name).read_bytes()
            digestlist[name]=sha(b)
            info=zipfile.ZipInfo("h52-train/"+name,(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,b,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist())!=len(names):
            raise ValueError("H52 ZIP inventory/CRC failure")
        if sha(z.read("h52-train/authoring_sft.jsonl"))!=H52_LABEL_SHA:
            raise ValueError("H52 annotations changed in package")
    shutil.copyfile(h50zip,folder/h50zip.name)
    meta=json.loads((v1/"dataset-metadata.json").read_text("utf8"))
    if meta["id"]!="abdulazizshehri/hala-gui-training-library":
        raise ValueError("Unexpected Kaggle destination")
    (folder/"dataset-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
    info={"schema":"hala.kaggle-library-version-two.v1","private_only":True,
          "v1_preserved_bit_identical":True,"h50_original_zip_sha256":V1_UPLOAD_SHA,
          "h50_original_label_sha256":V1_LABEL_SHA,"h52_unique_workflows":96,
          "h52_actual_browser_steps":288,"h52_distinct_original_screenshots":288,
          "h52_rejected_wrong_target_trials":4,"h52_language_rows":{"ar":144,"en":144},
          "h52_original_label_sha256":H52_LABEL_SHA,
          "h52_archive_sha256":sha(archive.read_bytes()),
          "h52_archive_bytes":archive.stat().st_size,
          "h52_file_sha256":digestlist,
          "external_sources_imported":False,"model_training_executed":False,
          "independent_model_benchmark_completed":False}
    (folder/"library-v2-manifest.json").write_text(
      json.dumps(info,sort_keys=True,indent=2)+"\n",encoding="utf8")
    (folder/"README.md").write_text(
      "# Hala original synthetic GUI training library (PRIVATE)\n\n"
      "Version 2 preserves the EXACT original H50 version-1 archive (64 unique "
      "original PNG, 384 single-step counterfactual annotations), then adds "
      "H52: 96 independent source-authored fictional workflows, 288 distinct "
      "new original state screenshots, and 288 browser-replayed correct "
      "steps with 4 wrong-target rejection probes. H52 split: Arabic 144, "
      "English 144. All source HTML+plans+browser receipts are saved with "
      "the H52 training archive; code and ledger live in TheHalfMoon/Hala.\n\n"
      "Zero real accounts, zero PHI, zero official benchmark TEST, zero "
      "unreviewed third-party screenshot uploads; NOT a trained model or "
      "evidence of improved accuracy. Metadata and item SHA256 audit in "
      "library-v2-manifest.json.\n",encoding="utf8")
    return info

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    info=build(Path(p.parse_args().output))
    print(json.dumps({"original_v1_verified":info["v1_preserved_bit_identical"],
       "h52_archive_sha256":info["h52_archive_sha256"],
       "h52_archive_bytes":info["h52_archive_bytes"],
       "original_workflows":info["h52_unique_workflows"],
       "training_steps":info["h52_actual_browser_steps"]}))

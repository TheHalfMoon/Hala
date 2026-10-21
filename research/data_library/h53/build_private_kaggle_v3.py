"""Create-only PRIVATE Kaggle Hala library v3 including genuine H Company TRAIN slider shard.

Never touches original H50/H52 archives, does not upload source screenshots to
PUBLIC GitHub, and does not claim any model finetuning or independent accuracy.
"""
from __future__ import annotations
import hashlib,json,shutil,zipfile
from pathlib import Path
from h53.dragon_train_shard import audit,sha
from research.data_library.h52.build_private_kaggle_v2 import build as v2_build
V1_SHA="783426a4f8ae10c2ce85fa6b9cc861f69857e64372e1a4dc8a4cf81b8053c37c"
V2_SHA="37d7514538d51fd76ca8edf56c14ecacb80235e6ad1a0a3835e307f455ad113b"
DEST="abdulazizshehri/hala-gui-training-library-private"
RETIRED_PUBLIC_SLUG="abdulazizshehri/hala-gui-training-library"

def build(root,out):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError("Private Kaggle v3 is strictly create-only")
    proof=audit(root)
    if proof["original_train_images"]<5 or proof["original_train_drag_annotations"]<15:
        raise ValueError("Original DragOn source incomplete")
    v2=out.parent/(out.name+"-v2")
    if v2.exists() or (out.parent/(v2.name+"-v1")).exists():
        raise FileExistsError("Private Kaggle v2-stage not clean")
    v2_build(v2)
    h50=v2/"hala-h50-train-original.zip"
    h52=v2/"hala-h52-multistep-train.zip"
    if sha(h50.read_bytes())!=V1_SHA or sha(h52.read_bytes())!=V2_SHA:
        raise ValueError("Historical first-party archives not byte-identical")
    out.mkdir(parents=True)
    for name in ("hala-h50-train-original.zip","hala-h52-multistep-train.zip",
                 "library-v2-manifest.json","dataset-metadata.json"):
        shutil.copyfile(v2/name,out/name)
    dataset_meta=out/"dataset-metadata.json"
    info=json.loads(dataset_meta.read_text("utf-8"))
    if info.get("id")!=RETIRED_PUBLIC_SLUG:
        raise ValueError("Historical H50/H52 builder identity mismatch")
    # Explicitly target the new authenticated PRIVATE dataset; never the
    # retired publicly exposed slug. The DragOn part is Apache-2.0, not CC0.
    info["id"]=DEST
    info["licenses"]=[{"name":"apache-2.0"}]
    info["description"]=(
        "PRIVATE mixed-origin Hala training library. H50/H52 first-party "
        "original synthetic train are CC0-1.0; H53 authentic Hcompany/DragOn "
        "TRAIN slider screenshot and drag targets are Apache-2.0. Preserve "
        "H Company attribution; no public DragOn EVAL/TEST, PHI or model weights."
    )
    dataset_meta.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    manifest=json.loads((root/"manifest.json").read_text("utf-8"))
    keys=[item["image_id"] for item in manifest["matched_source_entries"]]
    files=["train_drag.jsonl","manifest.json"]+[f"images/{key}.jpg" for key in keys]+[
      f"tasks/{key}.json" for key in keys]
    if len(files)!=2+len(keys)*2:raise ValueError("Shard file inventory invalid")
    contents={}
    for name in files:
        data=(root/name).read_bytes()
        if not data:raise ValueError("Empty official TRAIN member "+name)
        contents[name]=data
    bundle=out/"hala-h53-dragon-slider-original-train.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo("h53-dragon-slider-train/"+name,(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,contents[name],compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None or len(z.namelist())!=len(files):
            raise ValueError("Kaggle v3 ZIP corrupt")
        if sha(z.read("h53-dragon-slider-train/train_drag.jsonl"))!=proof["source_jsonl_sha256"]:
            raise ValueError("Original training labels changed in ZIP")
    saved=dict(schema="hala.private-training-library-v3.v1",kaggle_dataset=DEST,
        private_required=True,external_train_shard_uploaded=False,
        model_trained=False,independent_benchmark_qualified=False,
        no_eval_train_contamination=True,
        historical_h50_zip_sha256=V1_SHA,historical_h52_zip_sha256=V2_SHA,
        original_h52_v2_manifest_sha256=sha((out/"library-v2-manifest.json").read_bytes()),
        h53_drag_train_zip_sha256=sha(bundle.read_bytes()),h53_zip_size=bundle.stat().st_size,
        h53_original_drag_train_examples=proof["original_train_drag_annotations"],
        h53_unique_original_images=proof["original_train_images"],
        h53_label_sha256=proof["source_jsonl_sha256"],
        source="Hcompany/DragOn",source_split="train/slider.tar",
        upstream_revision=proof["source_revision"],source_license_label="Apache-2.0",
        full_provider_tar_content_sha256_qualified=False,
        all_visual_images_privacy_reviewed=False,
        source_files_sha256={name:sha(contents[name]) for name in sorted(files)})
    (out/"library-v3-manifest.json").write_text(json.dumps(saved,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# Hala PRIVATE training library v3\n\n"
        "Preserves original H50 and H52 source archives bit-for-bit. Adds a real "
        "subset of official Hcompany/DragOn **TRAIN** slider images and native "
        "drag start/end original bounding box targets, paired by exact image ID. "
        "Source repo: https://huggingface.co/datasets/Hcompany/DragOn "
        "(Apache-2.0, H Company credit). No eval/ or private TEST. "
        "This subset was fetched from two bounded HTTP 206 byte windows in "
        "the pinned original uncompressed 7GB TRAIN tar. Source TAR itself was "
        "NOT fully downloaded/rehash-verified. The range SHA256/source offsets, "
        "per-member original hashes and original raw JSON are included in "
        "h53 manifest, with no finetuned weights, trained model or benchmark score. "
        "Only the original shard lives in private Kaggle; public GitHub stores "
        "code and non-sensitive hash receipt, not raw third-party screenshots.\n",
        encoding="utf-8")
    return saved

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    d=build(a.source,a.output)
    print(json.dumps({"train_images":d["h53_unique_original_images"],
          "train_drag_tasks":d["h53_original_drag_train_examples"],
          "zip_sha256":d["h53_drag_train_zip_sha256"],
          "zip_size":d["h53_zip_size"],"private_only":True}))

"""Create-only strictly PRIVATE Kaggle original H57 TRAIN-only dataset staging.

Kaggle --private is required at creation. Never update an existing private
Kaggle dataset's metadata, to avoid repeating the prior public exposure.
Never commits external media or original task JSON to public GitHub.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import zipfile
from h57.import_train_multidomain import audit,SCHEMA,canonical,sha
DATASET="abdulazizshehri/hala-dragon-multidomain-train-private"
BUNDLE="hala-h57-original-drag-train-three-domains.zip"

def prepare(original,stage,repo_evidence):
    original,stage,repo_evidence=Path(original),Path(stage),Path(repo_evidence)
    if stage.exists() or repo_evidence.exists():
        raise FileExistsError("Source dataset stage and public receipt must be create-only")
    source=audit(original)
    manifest=json.loads((original/"manifest.json").read_text("utf8"))
    if (manifest["original_distinct_train_images"]<40 or
        manifest["original_train_drag_actions"]<400 or
        manifest["ordered_false"]<100 or
        manifest["ordered_true"]<100 or
        set(manifest["domains"])!={"sheet","slide_resize","text_highlight"}):
        raise ValueError("Insufficient real diverse original train data")
    stage.mkdir(parents=True)
    files=["manifest.json","train_drag_multidomain.jsonl"]
    for item in manifest["original_paired_image_task_members"]:
        domain,key=item["domain"],item["image_id"]
        files.extend([f"images/{domain}/{key}.jpg",f"tasks/{domain}/{key}.json"])
    if len(set(files))!=len(files):raise ValueError("Duplicate original member name")
    out=stage/BUNDLE
    with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,
                         allowZip64=True) as bundle:
        for name in sorted(files):
            data=(original/name).read_bytes()
            if not data:raise ValueError("Empty external original source member")
            info=zipfile.ZipInfo("h57-original-train/"+name,(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            bundle.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(out,"r") as bundle:
        if bundle.testzip() is not None:raise ValueError("Original train ZIP CRC mismatch")
        assert len(bundle.namelist())==len(files)
        recovered=bundle.read("h57-original-train/train_drag_multidomain.jsonl")
        if sha(recovered)!=manifest["training_label_sha256"]:
            raise ValueError("Train source has changed inside staged ZIP")
    metadata=dict(id=DATASET,title="Hala DragOn Multidomain Private TRAIN",
      subtitle="Genuine source-bound sheet, slide resize, and text highlighting drag labels",
      licenses=[dict(name="apache-2.0")],
      description=("PRIVATE Hala original source TRAIN data, Hcompany/DragOn pinned commit "
        "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5, Apache-2.0. "
        "Three additional native drag domains sheet, slide_resize, text_highlight; "
        "preserve original ordered true and false, original image/task bytes and SHA source "
        "offsets. Author-approved H Company data reuse asserted. Only small bounded byte "
        "range excerpts of original TRAIN TARs; upstream TAR 54GB total not independently "
        "rehashed, complete screenshot privacy review not completed. "
        "NO public DragOn eval or benchmark TEST, no trained model, no PHI source promised. "
        "Keep this Kaggle dataset PRIVATE always."))
    (stage/"dataset-metadata.json").write_bytes(canonical(metadata))
    private_manifest=dict(schema="hala.private.h57.hcompany-multidomain-dataset.v1",
        kaggle_dataset=DATASET,visibility_verified=False,
        mixed_licenses=False,all_original_media_private=True,
        original_source=manifest["original_source_repo"],
        source_revision=manifest["original_revision"],
        original_source_train_only=True,no_benchmark_answers=True,
        model_trained=False,blind_test_qualified=False,
        original_train_actions=source["actions"],original_unique_images=source["screens"],
        ordered_true=source["ordered_true"],ordered_false=source["ordered_false"],
        original_train_labels_sha256=source["original_train_sha256"],
        original_source_manifest_sha256=sha((original/"manifest.json").read_bytes()),
        private_train_zip_sha256=sha(out.read_bytes()),
        private_train_zip_bytes=out.stat().st_size,
        source_domain_summary={name:{k:val for k,val in info.items()
         if k!="rejected_matched_pairs"} for name,info in manifest["domains"].items()})
    (stage/"private-library-manifest.json").write_bytes(canonical(private_manifest))
    (stage/"README.md").write_text(
      "# H57 original DragOn TRAIN private\n\nAll media/JSON original training source is kept in"
      " this dataset, not public GitHub. Source Hcompany/DragOn, immutable commit "
      "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5, Apache-2.0. "
      "Do not publish original source media or move DEV/TEST into TRAIN. "
      "This sample is real source supervision, NOT model training or model evaluation.\n",
      encoding="utf8")
    # PUBLIC original-source hash-only receipt excludes image/task raw bytes and text.
    public=dict(schema="hala.h57.public.source-qualification-receipt.v1",
        original_source_repo=manifest["original_source_repo"],
        pinned_source_revision=manifest["original_revision"],
        train_domains=list(manifest["domains"]),
        original_train_screens=source["screens"],original_train_drag_actions=source["actions"],
        ordered_true=source["ordered_true"],ordered_false=source["ordered_false"],
        original_train_jsonl_sha256=source["original_train_sha256"],
        original_source_manifest_sha256=private_manifest["original_source_manifest_sha256"],
        private_kaggle_train_zip_sha256=private_manifest["private_train_zip_sha256"],
        kaggle_private_dataset=DATASET,kaggle_private_provider_confirmed=False,
        provider_declared_tar_sizes={k:v["total"] for k,v in
            __import__("h57.import_train_multidomain",fromlist=["DOMAINS"]).DOMAINS.items()},
        original_bounded_train_windows=manifest["original_range_receipts"],
        source_image_task_sha256=[{k:v for k,v in entry.items()
            if k in ("domain","image_id","image_sha256","task_sha256",
                     "original_image_source_offset","original_task_source_offset","task_count")}
             for entry in manifest["original_paired_image_task_members"]],
        original_source_domain_counters=private_manifest["source_domain_summary"],
        upstream_full_tars_verified=False,all_screenshots_visually_reviewed=False,
        model_training_executed=False,independent_model_inference_done=False,
        independently_blinded_model_benchmark_done=False,
        third_party_raw_images_committed_to_public_github=False)
    repo_evidence.write_bytes(canonical(public))
    return dict(dataset=DATASET,original_images=source["screens"],
                original_train_actions=source["actions"],
                ordered_true=source["ordered_true"],ordered_false=source["ordered_false"],
                private_zip_sha256=private_manifest["private_train_zip_sha256"],
                private_bytes=out.stat().st_size,provider_private_verified=False)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True)
    p.add_argument("--private-stage",required=True)
    p.add_argument("--public-hash-receipt",required=True)
    x=p.parse_args()
    print(json.dumps(prepare(x.original,x.private_stage,x.public_hash_receipt),sort_keys=True))

"""Create only a PRIVATE Kaggle SOURCE-MAP dataset, no original external media.

This metadata dataset has 116 planned TAR source windows and the MiniMax handoff,
NOT the 3,560,142 physically uploaded annotations. Never publish automatically.
"""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from h59.master_data_plan import check_plan,canonical

DATASET="abdulazizshehri/hala-million-source-map-private"
def stage(repo,output):
    repo,out=Path(repo),Path(output)
    if out.exists():raise FileExistsError("H59 Kaggle map stage must be new")
    plan=repo/"h59/full_source_windows.json"
    c=json.loads(plan.read_text("utf8"))
    checked=check_plan(c)
    out.mkdir(parents=True)
    shutil.copyfile(plan,out/"full_source_windows.json")
    for name in ("EXECUTION_MAP.md","MINIMAX_EXECUTION_PROMPT.md"):
        shutil.copyfile(repo/"h59"/name,out/name)
    source=hashlib.sha256((out/"full_source_windows.json").read_bytes()).hexdigest()
    status=dict(schema="hala.h59.kaggle-private-metadata-map.v1",dataset=DATASET,
        source_window_manifest_sha256=source,
        upstream_advertised_original_screenshots=checked["upstream_images"],
        upstream_advertised_drag_annotations=checked["upstream_tasks"],
        upstream_tar_total_bytes=checked["planned_bytes"],
        planned_windows=checked["planned_windows"],
        original_upstream_all_windows_downloaded=False,
        original_upstream_full_tar_sha_verified=False,
        original_full_3560142_training_rows_materialized=False,
        original_full_images_stored_private=False,
        model_training_executed=False,
        stored_existing_authentic_source_examples=2346,
        stored_existing_source_images=204,
        prior_private_kaggle_datasets=[
          "abdulazizshehri/hala-gui-training-library-private",
          "abdulazizshehri/hala-dragon-multidomain-train-private"],
        kaggle_metadata_dataset_private_provider_verified=False,
        status="SOURCE_MAP_ONLY_NOT_MILLION_SAMPLE_DATASET")
    (out/"SOURCE_STATUS.json").write_bytes(canonical(status))
    md=dict(id=DATASET,title="Hala 3.56M Source Map Private",
        subtitle="Pinned DragOn train source map; original media not included",
        licenses=[dict(name="apache-2.0")],
        description="PRIVATE metadata-only manifest for Hala/MiniMax. Describes 116 SHA-source-pinned train TAR byte windows covering 61.2GB of original Hcompany DragOn TRAIN upstream, 286012 upstream screenshots and 3560142 upstream annotations; these numbers are upstream availability only, NOT staged source examples. The only verified uploaded TRAIN remains 204 original images and 2346 original drag labels in TWO OTHER existing PRIVATE Kaggle datasets. No training performed. This is a source-ingestion plan. No donor JPEG/task images, model weights, Kaggle secrets, or official EVAL/TEST data included.")
    (out/"dataset-metadata.json").write_bytes(canonical(md))
    for p in out.rglob("*"):
        if p.is_file() and p.stat().st_size>2*1024*1024:
            raise ValueError("Metadata-only upload may not include any bulk private media")
    return dict(dataset=DATASET,manifest_sha256=source,
        planned_windows=checked["planned_windows"],media_files_uploaded=0,
        physical_million_training_examples_uploaded=False)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--repo",default=".")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(stage(args.repo,args.output),sort_keys=True))

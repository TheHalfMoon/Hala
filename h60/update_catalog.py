"""Update the living Hala data-library catalog with the measured H60 facts.

Only measured values are written. Anything still in flight is written as
in-progress, never as complete.
"""
from __future__ import annotations

import json
from pathlib import Path

CATALOG = Path("research/data_library/catalog.json")
catalog = json.loads(CATALOG.read_text("utf8"))
catalog["updated_utc"] = "2026-10-21"

for dataset in catalog["datasets"]:
    if dataset["id"] == "hcompany-dragon":
        dataset["advertised_upstream_images"] = 286012
        dataset["advertised_upstream_actions"] = 3560142
        dataset["measured_original_slider_images"] = 100000
        dataset["measured_original_slider_actions"] = 877709
        dataset["advertised_counts_superseded_by_measurement"] = True
        dataset["notes"] = (
            "Official upstream TRAIN only. The advertised 65,000 slider images / 571,350 actions "
            "undercount the pinned archive: the complete original slider TAR was streamed and its "
            "SHA-256 matches the official Hugging Face LFS digest, and it holds 100,000 original "
            "screenshots and 877,709 native drag actions with zero rejected rows."
        )

evaluation = next(
    item for item in catalog["evaluation_sources"] if item["id"] == "hala-h58-original-mixed-drag-public-dev"
)
evaluation["actual_qwen2b_real_inference_executed"] = True
evaluation["actual_qwen2b_real_inference_provider"] = "interactive_approved_free_colab_tesla_t4"
evaluation["actual_qwen2b_real_inference_cases"] = 24
evaluation["actual_qwen2b_strict_bare_json_valid"] = 0
evaluation["actual_qwen2b_tolerant_parse_valid"] = 16
evaluation["actual_qwen2b_ordered_semantics_valid"] = 16
evaluation["actual_qwen2b_geometrically_correct"] = 0
evaluation["actual_qwen2b_degenerate_start_equals_end"] = 6
evaluation["actual_qwen2b_out_of_range_points"] = 8
evaluation["actual_qwen2b_raw_attempts_sha256"] = (
    "21d5cc1e18f6c44c3d6805ef72e54ef6522592120b0ab5b9f7e26662388472da"
)
evaluation["genuine_neural_task_success_measured"] = True
evaluation["genuine_neural_task_success_count"] = 0
evaluation["independent_blind_test_completed"] = False
evaluation["model_weights_trained"] = False
evaluation["official_benchmark_comparison_qualified"] = False

evaluation55 = next(
    item for item in catalog["evaluation_sources"] if item["id"] == "hala-h55-bilingual-public-drag-dev"
)
evaluation55["actual_qwen2b_real_inference_executed"] = True
evaluation55["actual_qwen2b_real_inference_provider"] = "interactive_approved_free_colab_tesla_t4"
evaluation55["actual_qwen2b_real_inference_cases"] = 24
evaluation55["actual_qwen2b_strict_bare_json_valid"] = 0
evaluation55["actual_qwen2b_tolerant_parse_valid"] = 24
evaluation55["actual_qwen2b_ordered_semantics_valid"] = 24
evaluation55["actual_qwen2b_geometrically_correct"] = 0
evaluation55["actual_qwen2b_degenerate_start_equals_end"] = 21
evaluation55["actual_qwen2b_raw_attempts_sha256"] = (
    "17a00d3e0bb208d00617be7af5dc35a2f4fafb246f3d7217f6d6058ffdb95b12"
)
evaluation55["genuine_neural_task_success_measured"] = True
evaluation55["genuine_neural_task_success_count"] = 0
evaluation55["independent_blind_test_completed"] = False

catalog["source_maps"] = [
    item for item in catalog["source_maps"] if item["id"] != "hala-h60-full-corpus-dragon-train"
]
catalog["source_maps"].append(
    {
        "id": "hala-h60-full-corpus-dragon-train",
        "source": "Hcompany/DragOn",
        "source_revision": "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
        "source_kind": "original-train-full-corpus-ingest",
        "branch": "ps/h60-million-drag-train-ingest",
        "base_commit": "1746592e726c3f94c1272e6864072808688aa679",
        "planned_original_tar_windows": 116,
        "original_train_total_bytes": 61195154432,
        "advertised_upstream_train_images": 286012,
        "advertised_upstream_train_tasks": 3560142,
        "measured_original_slider_tar_sha256_verified": True,
        "measured_original_slider_tar_bytes": 7146830848,
        "measured_original_slider_images": 100000,
        "measured_original_slider_actions": 877709,
        "measured_slider_rejected_rows": 0,
        "remaining_domains_in_progress": ["sheet", "slide_resize", "text_highlight"],
        "private_provider_shards_published": 0,
        "private_provider_shard_readback_verified": 0,
        "actual_3560142_train_tasks_locally_parsed": False,
        "actual_3560142_train_tasks_private_kaggle_stored": False,
        "provider_private_dataset_namespace": "abdulazizshehri/hala-h60-dragon-train-<domain>-<index>-private",
        "provider_upload_bandwidth_measured_mb_s": 1.0,
        "provider_download_bandwidth_measured_mb_s": 32.6,
        "kaggle_gpu_compute_used": False,
        "kaggle_gpu_commercial_eligibility": "blocked_by_inspected_terms",
        "colab_free_t4_real_inference_executed": True,
        "colab_free_t4_real_lora_training_executed": True,
        "real_gpu_optimizer_updates": 0,
        "training_completed": False,
        "vercel_workflow_deployed": False,
        "independent_blind_test_completed": False,
    }
)

CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf8")
print(json.dumps({"updated": str(CATALOG), "bytes": CATALOG.stat().st_size}, sort_keys=True))

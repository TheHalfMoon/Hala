"""Aggregate-only sweep of original first-party v4 GUIs through a pinned processor.

This helper receives a verified per-image processor callback. It never fetches,
trains, scores a model, logs individual labels or stores synthetic screenshots.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

SPLITS = ("train", "dev")
DOMAINS = ("slider", "slide_resize", "text_highlight")
LANGUAGES = ("en", "ar")
SCHEMA = "hala.h426.full-image-processor-synthetic-sweep.v1"


def sweep_paired_processor(check: Callable, count_per_domain: int) -> dict:
    if type(count_per_domain) is not int or not 1 <= count_per_domain <= 64:
        raise ValueError("H426 full processor sweep must check 1..64 cases per domain")
    if not callable(check):
        raise ValueError("H426 verified processor callback required")
    counts = Counter()
    seen_image_sha = set()
    train_families = set()
    dev_families = set()
    for split in SPLITS:
        for domain in DOMAINS:
            for index in range(count_per_domain):
                pair = {}
                for language in LANGUAGES:
                    record = check(split, index, domain, language)
                    if (type(record) is not dict
                        or record.get("split") != split
                        or record.get("domain") != domain
                        or record.get("language") != language
                        or record.get("source_is_admitted_to_train") is not False
                        or record.get("new_model_trained") is not False
                        or record.get("model_release_qualified") is not False
                        or type(record.get("weighted_coordinate_tokens")) is not int
                        or record["weighted_coordinate_tokens"] < 2
                        or type(record.get("image_patch_tokens")) is not int
                        or record["image_patch_tokens"] <= 0
                        or type(record.get("synthetic_source_image_sha256")) is not str
                        or len(record["synthetic_source_image_sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in
                               record["synthetic_source_image_sha256"])
                        or type(record.get("synthetic_source_family")) is not str
                        or not record["synthetic_source_family"].startswith("invented-ui-family-")):
                        raise ValueError("H426 real processor sweep record unqualified")
                    pair[language] = record
                    counts[f"{split}/{domain}/{language}"] += 1
                sha = pair["en"]["synthetic_source_image_sha256"]
                family = pair["en"]["synthetic_source_family"]
                if (sha != pair["ar"]["synthetic_source_image_sha256"]
                    or family != pair["ar"]["synthetic_source_family"]
                    or sha in seen_image_sha):
                    raise ValueError("H426 real processor sweep image identity mismatch")
                seen_image_sha.add(sha)
                (train_families if split == "train" else dev_families).add(family)
    if train_families & dev_families:
        raise ValueError("H426 real processor sweep TRAIN/DEV source family overlap")
    return {
        "schema": SCHEMA,
        "generated_source_images_verified": len(seen_image_sha),
        "image_conditioned_actions_verified": sum(counts.values()),
        "by_split_domain_language": dict(sorted(counts.items())),
        "train_dev_family_overlap": False,
        "synthetic_source_admitted_to_train": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "new_model_trained": False,
        "independent_external_ood": False,
        "release_qualified": False,
    }

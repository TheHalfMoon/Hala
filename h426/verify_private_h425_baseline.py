"""Fail-closed qualification of owner-private H425 synthetic baseline aggregates.

A passing shape check does NOT prove the GPU/provider provenance; source SHA,
Kaggle result custody and original adapter hash must be verified separately.
"""
from __future__ import annotations

BASE = "Qwen/Qwen3-VL-2B-Instruct"
REV = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_SHA = "bc123aa521d58e35a205805759e38c8a6a4af4d62fc01ac3e118cc7aab7f4957"
RECEIPT_SHA = "14c48fcfe567c8c30465cc08ce323aba6841d4bb67bdfc30cc8750a3940a1057"
DOMAIN_SET = {"slider", "slide_resize", "text_highlight"}
FIELDS = {
    "images", "actions", "correct", "both_correct",
    "en_correct", "ar_correct", "en_end_only_error", "ar_end_only_error",
}
REQUIRED = {
    "schema","model","revision","actual_h425_adapter_sha256",
    "actual_h425_receipt_sha256","processor_pinned_public_files",
    "device","trainable_parameters","new_training_steps","source_role",
    "images","bilingual_actions","domains","raw_model_predictions_persisted",
    "model_weights_published","independent_external_ood",
    "source_rights_review_qualified","release_qualified",
}

def verify_aggregate(value: dict, *, count_per_domain: int = 2) -> dict:
    if type(count_per_domain) is not int or not 1 <= count_per_domain <= 32:
        raise ValueError("H426 actual model aggregate count must be 1..32")
    if type(value) is not dict or set(value) != REQUIRED:
        raise ValueError("H426 private aggregate receipt shape mismatch")
    exact = {
        "schema": "hala.h426.h425-pinned-readonly-synthetic-dev.v1",
        "model": BASE, "revision": REV,
        "actual_h425_adapter_sha256": MODEL_SHA,
        "actual_h425_receipt_sha256": RECEIPT_SHA,
        "processor_pinned_public_files": 8,
        "device": "Tesla T4",
        "trainable_parameters": 0, "new_training_steps": 0,
        "source_role": "firstparty_invented_author_visible_dev",
        "images": 3 * count_per_domain,
        "bilingual_actions": 6 * count_per_domain,
        "raw_model_predictions_persisted": False,
        "model_weights_published": False,
        "independent_external_ood": False,
        "source_rights_review_qualified": False,
        "release_qualified": False,
    }
    for name, expected in exact.items():
        got = value.get(name)
        if type(got) is not type(expected) or got != expected:
            raise ValueError("H426 private aggregate declared provenance/count mismatch")
    domains = value.get("domains")
    if type(domains) is not dict or set(domains) != DOMAIN_SET:
        raise ValueError("H426 aggregate source-domain completeness mismatch")
    for domain, stats in domains.items():
        if (type(stats) is not dict or set(stats) != FIELDS
            or any(type(x) is not int for x in stats.values())
            or stats["images"] != count_per_domain
            or stats["actions"] != 2 * count_per_domain):
            raise ValueError("H426 aggregate domain action cardinality mismatch")
        if (not 0 <= stats["correct"] <= 2 * count_per_domain
            or not 0 <= stats["both_correct"] <= count_per_domain
            or not 0 <= stats["en_correct"] <= count_per_domain
            or not 0 <= stats["ar_correct"] <= count_per_domain
            or stats["en_correct"] + stats["ar_correct"] != stats["correct"]
            or stats["both_correct"] > min(stats["en_correct"],
                                          stats["ar_correct"])
            or not 0 <= stats["en_end_only_error"] <= (count_per_domain -
                                                        stats["en_correct"])
            or not 0 <= stats["ar_end_only_error"] <= (count_per_domain -
                                                        stats["ar_correct"])):
            raise ValueError("H426 aggregate scored outcomes are internally inconsistent")
    return {
        "schema":"hala.h426.h425-synthetic-aggregate-contract-check.v1",
        "scored_images":3*count_per_domain,
        "scored_actions":6*count_per_domain,
        "all_three_source_domains_present":True,
        "provider_gpu_receipt_verified_here":False,
        "independent_external_ood":False,
        "release_qualified":False,
    }

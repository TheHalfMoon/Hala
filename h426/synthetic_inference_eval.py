"""Bounded read-only Qwen model evaluation on first-party invented H426 GUIs.

Caller supplies an ALREADY LOADED private model and a verified offline Qwen
processor. This module never downloads weights, uploads media, writes files,
trains, creates an optimizer, or reports independent OOD/release qualification.
"""
from __future__ import annotations

import os

from h426.synthetic_drag_curriculum import DOMAINS, generate
from h426.synthetic_h425_processor_bridge import pack_generated
from h426.synthetic_pair_eval import score_dev_pair

DOMAINS = tuple(DOMAINS)
LANGUAGES = ("en", "ar")
MAX_DEV_INDICES = 32


def validate_readonly_eval(count_per_domain: int):
    """Cheap pure-Python gates run BEFORE importing Torch or touching CUDA."""
    if type(count_per_domain) is not int or not 1 <= count_per_domain <= MAX_DEV_INDICES:
        raise ValueError("H426 synthetic model evaluation must be 1..32 DEV images per domain")
    if os.environ.get("HALA_H426_SYNTHETIC_READONLY_EVAL") != "1":
        raise ValueError("H426 synthetic model evaluation requires explicit opt-in")
    if (os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("H426 synthetic evaluation requires offline processor/model")
    return count_per_domain


def evaluate_already_loaded_model(model, processor, count_per_domain: int = 4):
    """Run *read-only* true model inference on synthetic, author-visible DEV.

    GPU and original private LoRA/model staging are the caller's responsibility.
    No model or processor path, adapter file, dataset or credentials is read.
    """
    validate_readonly_eval(count_per_domain)
    import torch
    if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "Tesla T4"):
        raise ValueError("H426 Qwen read-only model DEV evaluation requires owner FREE Kaggle T4")
    if type(processor).__name__ != "Qwen3VLProcessor":
        raise ValueError("H426 pinned official Qwen3VLProcessor instance required")
    if not hasattr(model, "generate") or not hasattr(model, "eval"):
        raise ValueError("H426 trained model instance unavailable")
    if not any(p.device.type == "cuda" for p in model.parameters()):
        raise ValueError("H426 supplied model is not resident on GPU")
    model.eval()
    source_binary_outcomes = []
    stats = {d: {"images": 0, "actions": 0, "correct": 0,
                 "both_correct": 0, "en_correct": 0, "ar_correct": 0,
                 "en_end_only_error": 0, "ar_end_only_error": 0}
             for d in DOMAINS}
    with torch.inference_mode():
        for domain in DOMAINS:
            for index in range(count_per_domain):
                source = generate("dev", index, domain)
                predictions = []
                for language in LANGUAGES:
                    item = pack_generated("dev", index, domain, language)
                    if item["source_sha256"] != source["image_sha256"]:
                        raise ValueError("H426 model inference image source mismatch")
                    user=[{"role":"user","content":[
                        {"type":"image"},
                        {"type":"text","text":item["request"]}]}]
                    header=processor.apply_chat_template(
                        user, tokenize=False, add_generation_prompt=True)
                    batch=processor(text=[header], images=[item["image"]],
                                    return_tensors="pt")
                    prompt_tokens=int(batch["input_ids"].shape[1])
                    if prompt_tokens <= 0:
                        raise ValueError("H426 model read-only visual prefix missing")
                    batch=batch.to("cuda:0")
                    generated=model.generate(
                        **batch, max_new_tokens=120, do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                    raw=processor.batch_decode(
                        generated[:,prompt_tokens:],skip_special_tokens=True
                    )[0].strip()
                    if type(raw) is not str or len(raw)>8192:
                        raise ValueError("H426 unbounded generated action text")
                    row=next(a for a in source["actions"] if a["language"]==language)
                    predictions.append({
                        "id":row["id"],"image_sha256":source["image_sha256"],
                        "raw_prediction":raw,
                    })
                    del batch, generated
                score=score_dev_pair(index,domain,predictions)
                # The only per-image information retained is matched synthetic
                # SHA and two BOOLEAN outcomes. Never preserve decoded text.
                source_binary_outcomes.append({
                    "image_sha256": source["image_sha256"],
                    "domain": domain,
                    "source_family": source["family"],
                    "languages": list(LANGUAGES),
                    "correct": [bool(score["per_language"][lang]["correct"])
                                for lang in LANGUAGES],
                })
                del predictions
                metrics=stats[domain]
                metrics["images"]+=1
                metrics["actions"]+=2
                metrics["correct"]+=score["correct_actions"]
                metrics["both_correct"]+=int(score["both_actions_correct"])
                for language in LANGUAGES:
                    language_score=score["per_language"][language]
                    metrics[language+"_correct"]+=int(language_score["correct"])
                    metrics[language+"_end_only_error"]+=int(
                        language_score["start_near"]
                        and not language_score["end_near"])
                # Predictions, prompts and image PNGs are never returned,
                # printed, uploaded or persisted by this evaluation routine.
    return {
        "schema":"hala.h426.invented-dev-readonly-model-eval.v1",
        "device":"Tesla T4",
        "images":3*count_per_domain,
        "bilingual_actions":6*count_per_domain,
        "synthetic_dev_author_visible":True,
        "model_or_processor_sha_authenticated_here":False,
        "trained_or_modified_model_here":False,
        "model_weights_saved_or_published":False,
        "independent_external_ood":False,
        "rights_review_qualified":False,
        "release_qualified":False,
        "domains":stats,
        "synthetic_per_image_binary_outcomes": source_binary_outcomes,
    }



def compare_readonly_synthetic_reports(baseline, candidate):
    """Compare sequential owner-model runs on SAME invented DEV screenshots.

    Runs may load one model at a time on a single free T4, avoiding simultaneous
    two-model VRAM use. Results MUST include only boolean outcomes (no raw text).
    No real inference, attestation, training or source admission occurs here.
    """
    from h426.paired_study_statistics import summarize_matched_images
    for report in (baseline, candidate):
        if (type(report) is not dict
            or report.get("schema") != "hala.h426.invented-dev-readonly-model-eval.v1"
            or report.get("synthetic_dev_author_visible") is not True
            or report.get("trained_or_modified_model_here") is not False
            or report.get("model_weights_saved_or_published") is not False
            or report.get("independent_external_ood") is not False
            or report.get("release_qualified") is not False
            or type(report.get("synthetic_per_image_binary_outcomes")) is not list):
            raise ValueError("H426 only source-authenticated invented DEV reports can be paired")
    old = baseline["synthetic_per_image_binary_outcomes"]
    new = candidate["synthetic_per_image_binary_outcomes"]
    if len(old) != len(new) or not old:
        raise ValueError("H426 original and candidate source DEV sizes differ")
    def keyed(items):
        registry = {}
        for r in items:
            if (type(r) is not dict or set(r) != {
                "image_sha256", "domain", "source_family",
                "languages", "correct",
            } or type(r.get("image_sha256")) is not str
                or type(r.get("domain")) is not str
                or type(r.get("source_family")) is not str
                or r.get("languages") != ["en", "ar"]
                or type(r.get("correct")) is not list
                or len(r["correct"]) != 2
                or any(type(x) is not bool for x in r["correct"])):
                raise ValueError("H426 synthetic DEV source identity/outcome drift")
            k = (r["domain"], r["image_sha256"])
            if k in registry:
                raise ValueError("H426 duplicated invented DEV source screenshot")
            registry[k] = r
        return registry
    left, right = keyed(old), keyed(new)
    if left.keys() != right.keys():
        raise ValueError("H426 baseline and candidate did not see same screenshots")
    paired=[]
    for k in sorted(left):
        before,after=left[k],right[k]
        if (before["source_family"] != after["source_family"]
            or before["languages"] != after["languages"]):
            raise ValueError("H426 baseline/candidate synthetic family or language drift")
        paired.append({
            "image_sha256":k[1], "domain":k[0],
            "source_family":before["source_family"],
            "languages":before["languages"],
            "baseline_correct":before["correct"],
            "candidate_correct":after["correct"],
        })
    comparison=summarize_matched_images(paired)
    comparison["evaluation_source"] = "invented_author_visible_not_external_ood"
    comparison["model_predictions_verified_here"] = False
    return comparison

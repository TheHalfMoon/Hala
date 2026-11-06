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
    }

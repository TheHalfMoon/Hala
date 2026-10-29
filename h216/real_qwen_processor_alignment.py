"""H216 real Qwen3-VL processor-only multimodal SFT alignment preflight.

The only permitted live asset is the owner-cached, PINNED Qwen3-VL
processor/tokenizer. No Qwen model weights, model.generate, optimizer,
private media, TRAIN admission, sealed DEV or consumed historical OOD.
An observed processor/token prefix failure MUST remain a failure.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, assistant_completion_mask, parse_actions,
)
from h212.procedural_native_gui_candidates import (
    COUNT, research_multimodal_messages,
)
from h213.frozen_candidate_reproduction_gate import verify_checked_in_source

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_PROCESSOR_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
CASES = (0, 1, 64, 65)
MAX_PREFIX_TOKENS = 18000
MAX_FULL_TOKENS = 18500


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H216 real processor alignment blocked: "+reason)


def token_ids(batch, *, what: str) -> list[int]:
    require("input_ids" in batch and len(batch["input_ids"]) == 1,
            what + " must have one tokenized task row")
    source = batch["input_ids"][0]
    require(callable(getattr(source,"tolist",None)),
            what + " must expose integer token tensor")
    ids = source.tolist()
    require(type(ids) is list and len(ids) >= 2
            and all(type(t) is int and t >= 0 for t in ids),
            what + " must have valid positive token IDs")
    return ids


def visual_tensor_receipt(prompt, full) -> int:
    """Check actual visual pixel inputs are identical in both tokenizations."""
    import torch
    require("pixel_values" in prompt and "pixel_values" in full,
            "model-visible image tensors required")
    pi, fi = prompt["pixel_values"], full["pixel_values"]
    require(callable(getattr(pi,"numel",None))
            and callable(getattr(fi,"numel",None))
            and 0 < int(pi.numel()) == int(fi.numel())
            and torch.equal(pi,fi),
            "real image pixels changed during supervised tokenization")
    for key in ("image_grid_thw",):
        require(key in prompt and key in full
                and torch.equal(prompt[key],full[key]),
                "visual image grid changed in SFT answer")
    return int(pi.numel())


def audit_sample(processor, case_index: int) -> dict:
    """Real PIL → actual HF processor; strictly no generative/model calls."""
    require(type(case_index) is int and 0 <= case_index < COUNT,
            "only bounded, self-authored H212 source examples")
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "H199 TRAIN admission is not approved")
    question, answer = research_multimodal_messages(case_index)
    target = answer[-1]["content"][0]["text"]
    native = parse_actions(target, training_target=True)
    require(native["actions"][0]["kind"] == "click",
            "only source-computed visible click label allowed")
    args = dict(
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    prompt = processor.apply_chat_template(
        question, add_generation_prompt=True, **args)
    full = processor.apply_chat_template(
        answer, add_generation_prompt=False, **args)
    prefix = token_ids(prompt,what="prompt")
    sequence = token_ids(full,what="SFT target")
    require(0 < len(prefix) <= MAX_PREFIX_TOKENS
            and len(prefix)+5 <= len(sequence) <= MAX_FULL_TOKENS,
            "multimodal token lengths outside explicit source bounds")
    pixnum = visual_tensor_receipt(prompt,full)
    require(sequence[:len(prefix)] == prefix,
            "Qwen3-VL tokenizer prefix changed between prompt and answer")
    supervised = assistant_completion_mask(prefix,sequence)
    require(supervised[:len(prefix)] == [-100]*len(prefix)
            and supervised[len(prefix):] == sequence[len(prefix):],
            "loss covers user/vision or masks completion")
    texts = processor.batch_decode(
        [sequence[len(prefix):]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    require(type(texts) is list and len(texts)==1
            and type(texts[0]) is str
            and parse_actions(texts[0],training_target=True) == native,
            "assistant continuation does not strictly decode to target click")
    return {
        "case_index":case_index,
        "prompt_tokens":len(prefix),
        "full_tokens":len(sequence),
        "supervised_tokens":len(sequence)-len(prefix),
        "masked_prompt_tokens":len(prefix),
        "vision_tensor_elements":pixnum,
        "input_prefix_exact":True,
        "vision_values_identical":True,
        "assistant_native_click_roundtrip":True,
        "model_loaded":False,
        "training_step_performed":False,
    }


def verify_cached_processor_root(root: Path) -> None:
    require(isinstance(root,Path) and root.is_absolute()
            and root.name == QWEN_PROCESSOR_REV,
            "exact offline cached Qwen3-VL revision required")
    require(root.is_dir() and not root.is_symlink(),
            "processor cache snapshot unavailable")
    for filename in ("config.json","preprocessor_config.json",
                     "tokenizer.json","tokenizer_config.json"):
        require((root/filename).is_file(),
                "pinned offline tokenizer/processor file missing")
    require(os.environ.get("HF_HUB_OFFLINE")=="1"
            and os.environ.get("TRANSFORMERS_OFFLINE")=="1",
            "disallow any online cache/model weight fetch")
    # H218: assert SHA-256 of the exact eight previously observed PUBLIC
    # Qwen3-VL processor/tokenizer bytes before any HF processor import.
    # Neither expected files nor digests can be overridden from the CLI.
    from h218.pinned_qwen_processor_bytes import verify_pinned_snapshot
    receipt = verify_pinned_snapshot(root)
    require(receipt["verified_files"] == 8,
            "incomplete original public Qwen processor byte receipt")


def run_real_processor_audit(root: Path, *, all_research: bool = False) -> list[dict]:
    verify_cached_processor_root(root)
    verified = verify_checked_in_source()
    require(verified["candidate_cases_checked"] == COUNT
            and verified["rgba_sha_matches"] == COUNT
            and not verified["official_train_admitted"],
            "H212 frozen source fingerprints not safely qualified")
    from transformers import AutoProcessor, Qwen3VLProcessor
    proc = AutoProcessor.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False,
    )
    require(type(proc) is Qwen3VLProcessor,
            "loaded processor is not the exact Qwen3VLProcessor class")
    require(getattr(proc,"apply_chat_template",None) is not None
            and getattr(proc,"batch_decode",None) is not None,
            "real Qwen3-VL processor not available")
    print("H216_PINNED_QWEN3VL_REAL_PROCESSOR_LOADED_NO_WEIGHTS",flush=True)
    result=[]
    selected = range(COUNT) if all_research else CASES
    for index in selected:
        receipt=audit_sample(proc,index)
        result.append(receipt)
        if not all_research or (index+1) % 16 == 0:
            print("H216_REAL_PROCESSOR_CHECKPOINT",
                  "cases_passed="+str(len(result)),
                  "prompt_tokens="+str(receipt["prompt_tokens"]),
                  "assistant_tokens="+str(receipt["supervised_tokens"]),
                  "vision_elements="+str(receipt["vision_tensor_elements"]),
                  "prefix_exact=true click_roundtrip=true",
                  flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description="Qwen3-VL processor only, no weights")
    ap.add_argument("--local-processor-snapshot",required=True,type=Path)
    ap.add_argument("--all-research",action="store_true",
                    help="verify all 96 in-memory H212 candidates, never train")
    ns=ap.parse_args()
    try:
        receipts=run_real_processor_audit(
            ns.local_processor_snapshot, all_research=ns.all_research,
        )
    except Exception:
        # No raw user tokens or images in stderr, no fake PASS.
        raise SystemExit("H216_REAL_QWEN_PROCESSOR_ALIGNMENT_BLOCKED")
    print("H216_QWEN3VL_H212_REAL_PROCESSOR_ALIGNMENT_PASS",
          "synthetic_cases="+str(len(receipts)),
          "source_rgba_verified=96",
          "model_calls=0 optimizer_steps=0 official_train=0 official_dev=0 official_ood=0",
          flush=True)


if __name__ == "__main__":
    main()

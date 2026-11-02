"""H361 pinned Qwen3-VL processor parity for H199 original candidate rows.

NOT a trainer. This validates native screenshot/image + assistant completion
processor semantics, only AFTER H359 audits the source-only candidate bundle.
No model weights, optimizer, adapter, TRAIN admission, or blind OOD.
The owner PC currently lacks torch/transformers; CPU mock tests are NOT
evidence of real pinned Qwen processor parity.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, assistant_completion_mask,
    parse_actions, source_task_prompt,
    training_messages_for_synthetic_contract,
)
from h216.real_qwen_processor_alignment import (
    QWEN_PROCESSOR_REV, MAX_FULL_TOKENS, MAX_PREFIX_TOKENS,
    token_ids, visual_tensor_receipt, verify_cached_processor_root,
)
from h359.native_candidate_bridge import (
    SOURCES, PREFIX, audit_review_bundle, safe_regular,
)
from training.artifacts import strict_json

ROLE = "H361_SOURCE_CANDIDATE_PROCESSOR_ONLY_NOT_TRAIN"
SCHEMA = "hala.h361.original-native-processor-parity.v1"
# First two bilingual cases in each action phase; fixed, publicly exposed.
# This is source-only processor qualification, NEVER blind OOD/DEV.
SELECTED = {
    "controlgarden-single-app": (0, 1, 2, 3),
    "noterelay-original-multistep": tuple(range(8)),
}
MODEL_CALLS = 0


def refuse(ok: bool):
    if not ok:
        raise ValueError("H361_ORIGINAL_PROCESSOR_PARITY_NOT_QUALIFIED")


def _candidate_rows(root: Path, family: str) -> list[dict]:
    # Before opening any PIL media or loading any processor:
    # audit all native model actions, original source byte SHA and media.
    report = audit_review_bundle(root, family)
    refuse(report["family"] == family and
           report["official_train_admitted"] is False and
           report["can_train"] is False and
           report["blind_ood"] is False)
    source = safe_regular(root, "candidate-rows.jsonl")
    rows = [strict_json(line.decode("utf-8"))
            for line in source.read_bytes().splitlines()]
    refuse(len(rows) == SOURCES[family]["records"])
    return rows


def real_processor_messages(processor, root: Path, family: str) -> list[dict]:
    """Tokenize only 4+8 original candidate samples using supplied processor.

    The supplied object MUST be a separately verified pinned real Qwen
    processor in production. Fake processors are unit-test policy plumbing
    only, NOT a real tokenizer qualification.
    """
    refuse(not ADMITTED_TRAIN_MANIFEST_SHAS)
    refuse(isinstance(root, Path) and root.is_dir() and family in SELECTED)
    rows = _candidate_rows(root, family)
    indexes = SELECTED[family]
    expected = SOURCES[family]
    receipts = []
    from PIL import Image

    for index in indexes:
        row = rows[index]
        refuse(row["family"] == family
               and row["original_source_sha256"] == expected["source_sha"]
               and type(row["prompt"]) is str
               and row["prompt"].startswith(PREFIX))
        task = row["prompt"][len(PREFIX):]
        refuse(source_task_prompt(task) == row["prompt"])
        action = parse_actions(row["completion"], training_target=True)
        path = safe_regular(root, row["image_path"])
        refuse(path.stat().st_size <= 4_000_000)
        raw = path.read_bytes()
        refuse(sha256(raw).hexdigest() == row["image_sha256"])
        with Image.open(BytesIO(raw)) as original:
            refuse(original.format == "PNG" and original.size == (960,720))
            image = original.convert("RGB")
        question, answer = training_messages_for_synthetic_contract(
            task, row["completion"])
        # Exact SAME PIL object for prompt and answer; never a distinct
        # re-decoding/re-rendering of the snapshot to prevent image drift.
        question[0]["content"][0]["image"] = image
        answer[0]["content"][0]["image"] = image
        refuse(question[0]["content"][0]["image"]
               is answer[0]["content"][0]["image"])
        args = {"tokenize": True, "return_dict": True,
                "return_tensors": "pt"}
        prompt_batch = processor.apply_chat_template(
            question, add_generation_prompt=True, **args)
        target_batch = processor.apply_chat_template(
            answer, add_generation_prompt=False, **args)
        prefix = token_ids(prompt_batch, what="H361 prompt")
        full = token_ids(target_batch, what="H361 full")
        refuse(0 < len(prefix) <= MAX_PREFIX_TOKENS
               and len(prefix) + 5 <= len(full) <= MAX_FULL_TOKENS
               and full[:len(prefix)] == prefix)
        pixels = visual_tensor_receipt(prompt_batch, target_batch)
        mask = assistant_completion_mask(prefix, full)
        refuse(len(mask) == len(full)
               and mask[:len(prefix)] == [-100]*len(prefix)
               and mask[len(prefix):] == full[len(prefix):]
               and all(value >= 0 for value in mask[len(prefix):]))
        decoded = processor.batch_decode(
            [full[len(prefix):]], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        refuse(type(decoded) is list and len(decoded) == 1
               and type(decoded[0]) is str
               and parse_actions(decoded[0], training_target=True) == action)
        receipts.append({
            "language": row["language"],
            "kind": action["actions"][0]["kind"],
            "image_tensor_elements": pixels,
            "prompt_tokens": len(prefix),
            "completion_tokens": len(full) - len(prefix),
            "prefix_exact": True, "assistant_only_mask": True,
            "image_identical": True, "action_roundtrip": True,
            "model_calls": 0, "optimizer_steps": 0,
            "training_authorized": False, "blind_ood": False,
        })
    refuse(len(receipts) == len(indexes))
    return receipts


def run_pinned_offline_candidate_processor(
    snapshot: Path, datasets: dict[str, Path]
) -> dict:
    """Real CPU tokenizer/images only. Never fetch model weights, or train."""
    refuse(not ADMITTED_TRAIN_MANIFEST_SHAS
           and type(datasets) is dict
           and set(datasets) == set(SELECTED)
           and all(isinstance(path, Path) for path in datasets.values()))
    # Verify all PUBLIC candidate sources and bytes before HF import.
    for family, root in datasets.items():
        _candidate_rows(root, family)
    verify_cached_processor_root(snapshot)
    from transformers import AutoProcessor, Qwen3VLProcessor
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    refuse(type(processor) is Qwen3VLProcessor)
    findings = [
        receipt
        for family, root in datasets.items()
        for receipt in real_processor_messages(processor, root, family)
    ]
    refuse(len(findings) == 12
           and sum(r["language"] == "ar" for r in findings) == 6
           and sum(r["language"] == "en" for r in findings) == 6
           and {r["kind"] for r in findings} == {"click","type_text"})
    # Never output raw tasks, image bytes, prompt text, gold, tokens,
    # selected source IDs, source paths, model logits or model weights.
    return {
        "schema": SCHEMA, "role": ROLE,
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "processor_revision": QWEN_PROCESSOR_REV,
        "source_candidate_processor_rows": len(findings),
        "languages": {"en": 6, "ar": 6},
        "actions": {"click": 10, "type_text": 2},
        "processor_image_and_prompt_parity": True,
        "model_weights_loaded": False,
        "optimizer_steps": 0,
        "official_train_admitted": False,
        "independent_rights_verified": False,
        "new_blind_ood": False, "can_train": False,
        "can_release": False,
    }

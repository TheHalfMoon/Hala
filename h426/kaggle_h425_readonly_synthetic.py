"""Owner-private read-only H425 baseline inference on H426 invented DEV.

Bundled into a Kaggle PRIVATE script by build_private_readonly_kaggle.py.
Never commit outputs or upload adapter/input bytes to the public repository.
This measures already-trained H425 only. It does NOT train H426 or prove OOD.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

BASE = "Qwen/Qwen3-VL-2B-Instruct"
REV = "89644892e4d85e24eaac8bacfd4f463576704203"
H425_WEIGHT_SHA = "bc123aa521d58e35a205805759e38c8a6a4af4d62fc01ac3e118cc7aab7f4957"
H425_RECEIPT_SHA = "14c48fcfe567c8c30465cc08ce323aba6841d4bb67bdfc30cc8750a3940a1057"
RESULT = Path("/kaggle/working/h426-owner-private-h425-invented-dev-aggregate.json")
COUNT = 2


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _gpu_preflight():
    if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or RESULT.exists()):
        raise ValueError("H426 demands owner Kaggle and create-only private result")
    import torch
    if (not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "Tesla T4"):
        raise ValueError("H426 baseline must use owner free Tesla T4")
    return torch


def _trusted_processor_snapshot():
    """Download ONLY public immutable processor files, then verify all bytes."""
    from huggingface_hub import snapshot_download
    from h218.pinned_qwen_processor_bytes import (
        PINNED_PUBLIC_FILES, verify_pinned_snapshot,
    )
    source = Path(snapshot_download(
        repo_id=BASE, revision=REV,
        allow_patterns=list(PINNED_PUBLIC_FILES),
    ))
    # Make a true 8-file, symlink-free copy; the HF cache itself is mutable.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix=".h426-public-processor-",
                            dir="/kaggle/working") as tmp:
        pinned = Path(tmp)/REV
        pinned.mkdir()
        for name, (_, expected) in sorted(PINNED_PUBLIC_FILES.items()):
            payload = (source/name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("H426 pinned processor public bytes changed")
            (pinned/name).write_bytes(payload)
        verified = verify_pinned_snapshot(pinned)
        if verified["verified_files"] != 8:
            raise ValueError("H426 incomplete pinned Qwen image processor")
        from h387.verified_processor_handoff import verified_processor_handoff
        with verified_processor_handoff(pinned) as immutable:
            yield immutable


def run():
    torch = _gpu_preflight()
    # Version-pinned, same as original H425 GPU evaluation; no model training.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.57.3", "peft==0.18.0",
        "accelerate==1.12.0", "Pillow==11.3.0",
    ], check=True, timeout=360)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import transformers
    from peft import PeftModel
    if transformers.__version__ != "4.57.3":
        raise ValueError("H426 pinned H425 processor version changed")

    # Reuse the original H425 source-qualified child (complete receipt checks).
    parent = Path(__file__).resolve().parents[1]
    legacy = runpy.run_path(
        str(parent/"h425/kaggle/h425_h421_paired_eval.py"),
        run_name="h426_readonly_h425_verified_helpers",
    )
    trained_receipt, receipt_sha, weight, config = (
        legacy["h425_true_trained_child"]()
    )
    if (receipt_sha != H425_RECEIPT_SHA
        or sha_file(weight) != H425_WEIGHT_SHA
        or trained_receipt.get("model_weights_sha256") != H425_WEIGHT_SHA):
        raise ValueError("H426 refusing H425 child adapter provenance drift")

    from contextlib import contextmanager
    processor_ctx = contextmanager(_trusted_processor_snapshot)
    with processor_ctx() as snapshot, TemporaryDirectory(
        prefix=".h426-private-h425-adapter-", dir="/kaggle/working",
    ) as temporary:
        # Keep the source immutable throughout the private source read.
        path = Path(temporary)
        shutil.copyfile(weight, path/"adapter_model.safetensors")
        shutil.copyfile(config, path/"adapter_config.json")
        if sha_file(path/"adapter_model.safetensors") != H425_WEIGHT_SHA:
            raise ValueError("H426 private H425 staged adapter SHA changed")

        processor = AutoProcessor.from_pretrained(
            str(snapshot), trust_remote_code=False,
            local_files_only=True, use_fast=False,
        )
        if type(processor).__name__ != "Qwen3VLProcessor":
            raise ValueError("H426 official pinned image processor missing")
        base = AutoModelForImageTextToText.from_pretrained(
            BASE, revision=REV, dtype=torch.float16,
            trust_remote_code=False, attn_implementation="sdpa",
        ).to("cuda:0")
        model = PeftModel.from_pretrained(
            base, str(path), is_trainable=False,
        )
        model.eval()
        for parameter in model.parameters():
            if parameter.requires_grad:
                raise ValueError("H426 read-only model has trainable parameters")
        os.environ["HALA_H426_SYNTHETIC_READONLY_EVAL"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from h426.synthetic_inference_eval import evaluate_already_loaded_model
        report = evaluate_already_loaded_model(
            model, processor, count_per_domain=COUNT,
        )
        if (report["images"] != 3 * COUNT
            or report["bilingual_actions"] != 6 * COUNT
            or report["trained_or_modified_model_here"] is not False
            or report["release_qualified"] is not False):
            raise ValueError("H426 baseline outcome contract invalid")
        # Drop individual synthetic SHA/outcomes. Aggregates only.
        safe = {
            "schema": "hala.h426.h425-pinned-readonly-synthetic-dev.v1",
            "model": BASE,
            "revision": REV,
            "actual_h425_adapter_sha256": H425_WEIGHT_SHA,
            "actual_h425_receipt_sha256": H425_RECEIPT_SHA,
            "processor_pinned_public_files": 8,
            "device": "Tesla T4",
            "trainable_parameters": 0,
            "new_training_steps": 0,
            "source_role": "firstparty_invented_author_visible_dev",
            "images": report["images"],
            "bilingual_actions": report["bilingual_actions"],
            "domains": report["domains"],
            "raw_model_predictions_persisted": False,
            "model_weights_published": False,
            "independent_external_ood": False,
            "source_rights_review_qualified": False,
            "release_qualified": False,
        }
        raw = json.dumps(safe, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False).encode("utf8")
        with RESULT.open("xb") as output:
            output.write(raw)
        print("H426_H425_PRIVATE_SYNTHETIC_BASELINE_AGGREGATE",
              json.dumps(safe, sort_keys=True, ensure_ascii=False), flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()

"""H173 frozen unseen-app OOD GPU proposals, NO gold, NO training.

This is an intentionally UNQUALIFIED runner until a NEW original source
passes 32/32 exact independent pixel replay and its ZIP SHA is pinned.
Never use H172 consumed DEV or prior OOD to fill in missing inputs.
"""
import hashlib
import importlib.metadata
import io
import json
import os
import zipfile
from pathlib import Path

from h132.grounder_contract import MODEL_ID, MODEL_SHA
from h164.native_action_contract import parse_action
from h166.vision_probe import MAX_PEAK_GIB, require_free_t4
from h167.combined_probe import (
    LANGUAGE_WEIGHTS_SHA,
    admit_frozen_language_directory,
    activate_frozen_language_and_trainable_vision,
)
from h172.infer_gpu import admit_vision as admit_h171_frozen_vision
from h173.pack_ood import SCHEMA as OOD_SCHEMA
from h173.source_preflight import OOD_COUNT, preregister_provisional_cases
from h173.stable_capture import original_visual_hash

# Fill ONLY after a brand-new 32/32 original OOD screenshot replay and
# private gold-free ZIP. Not a placeholder for an older DEV or prior OOD.
OOD_ZIP_SHA = "UNQUALIFIED"
MODEL_PROPOSALS_SCHEMA = "hala.h173.frozen-paired-unseen-original-ood-predictions.v1"


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_unseen_ood(path):
    """Exactly 32 new native source screenshots, no owner gold or old OOD."""
    if len(OOD_ZIP_SHA) != 64 or sha_file(path) != OOD_ZIP_SHA:
        raise ValueError("H173 pinned NEW unseen original-app OOD ZIP required")
    frozen = preregister_provisional_cases()
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        files = set(entries)
        expected_files = {"manifest.json"} | {
            "images/" + str(row["index"]) + ".png" for row in frozen
        }
        if len(entries) != len(expected_files) or files != expected_files:
            raise ValueError(
                "H173 only 32 unique original OOD PNGs and a single manifest accepted"
            )
        manifest = json.loads(archive.read("manifest.json"))
        rows = manifest.get("private_cases")
        if (
            manifest.get("schema") != OOD_SCHEMA
            or manifest.get("split") != "OOD"
            or manifest.get("count") != OOD_COUNT
            or manifest.get("never_previously_used_original_app_families")
            is not True
            or manifest.get("previous_train_dev_ood_excluded") is not True
            or manifest.get("owner_gold_uploaded") is not False
            or not isinstance(rows, list)
            or len(rows) != OOD_COUNT
            or len(manifest.get("owner_source_manifest_sha256", "")) != 64
            or len(manifest.get("independent_rgba_replay_receipt_sha256", ""))
            != 64
        ):
            raise ValueError("H173 original unseen-app provenance incomplete")
        seen = set()
        visible = set()
        for row, expected in zip(rows, frozen):
            if (
                any(
                    row.get(key) != value
                    for key, value in expected.items()
                    if key != "split"
                )
                or row.get("split") != "OOD"
                or row.get("image") != f"images/{row['index']}.png"
                or row.get("image_sha256") in seen
                or hashlib.sha256(archive.read(row["image"])).hexdigest()
                != row.get("image_sha256")
                or not isinstance(row.get("instruction"), str)
                or len(row["instruction"]) < 8
                or any(key in row for key in (
                    "training_supervision", "reference_point", "owner_gold",
                    "owner_reference_actions", "owner_native_action", "gold"
                ))
            ):
                raise ValueError("H173 unseen OOD image/task altered or gold leaked")
            visual = original_visual_hash(archive.read(row["image"]))
            if visual in visible:
                raise ValueError("H173 visually duplicate original app OOD image")
            visible.add(visual)
            seen.add(row["image_sha256"])
        if len(seen) != OOD_COUNT:
            raise ValueError("H173 incomplete unique original OOD screenshots")
    return rows


def run(ood_zip, frozen_language, frozen_vision, output):
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from h164.train_gpu import predict

    output = Path(output)
    if (
        output.exists()
        or not str(output).startswith("/kaggle/working/")
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH") != "1"
    ):
        raise ValueError("H173 owner-private FREE Kaggle T4 only")
    versions = {
        key: importlib.metadata.version(key).split("+")[0]
        for key in ("torch", "transformers", "peft", "Pillow")
    }
    gpu = require_free_t4(torch, versions)
    rows = admit_unseen_ood(ood_zip)
    language = admit_frozen_language_directory(
        frozen_language, Path("/kaggle/tmp/h173-frozen-h164-language")
    )
    visual = admit_h171_frozen_vision(frozen_vision)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_SHA, trust_remote_code=False
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_SHA, torch_dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=False
    ).to("cuda:0")
    model = PeftModel.from_pretrained(
        base, str(language), adapter_name="language", is_trainable=False
    )
    model.load_adapter(str(visual), adapter_name="vision", is_trainable=False)
    model.eval()
    predictions = []
    counts = {"before": 0, "after": 0}
    torch.cuda.reset_peak_memory_stats(0)
    with zipfile.ZipFile(ood_zip) as archive:
        for role in ("before", "after"):
            if role == "before":
                model.set_adapter("language")
                if list(model.active_adapters) != ["language"]:
                    raise ValueError("H173 frozen baseline incorrectly composed")
            else:
                active = activate_frozen_language_and_trainable_vision(model)
                if active != ["language", "vision"]:
                    raise ValueError("H173 H164+H171 dual LoRA not active")
            for row in rows:
                with Image.open(io.BytesIO(archive.read(row["image"]))) as im:
                    picture = im.convert("RGB")
                if picture.size != (960, 720):
                    raise ValueError("H173 original image no longer 960x720")
                raw = predict(processor, model, torch, picture, row)
                parsed = parse_action(raw)
                predictions.append({
                    "role": role,
                    "case_index": row["index"],
                    "image_sha256": row["image_sha256"],
                    "app": row["app"],
                    "language": row["language"],
                    "action_type": row["action_type"],
                    "raw": raw,
                    "parsed": parsed,
                })
                counts[role] += 1
                if len(predictions) % 8 == 0:
                    print("H173_NEW_UNSEEN_FROZEN_MODEL", len(predictions), flush=True)
    peak = torch.cuda.max_memory_allocated(0) / 2**30
    if (
        counts != {"before": OOD_COUNT, "after": OOD_COUNT}
        or len(predictions) != OOD_COUNT * 2
        or not 0 < peak <= MAX_PEAK_GIB
    ):
        raise ValueError("H173 real FREE T4 frozen OOD evidence incomplete")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": MODEL_PROPOSALS_SCHEMA,
        "status": "H173_PRIVATE_64_REAL_GPU_UNSEEN_OOD_PROPOSALS_COMPLETE",
        "source_ood_zip_sha256": OOD_ZIP_SHA,
        "frozen_language_sha256": LANGUAGE_WEIGHTS_SHA,
        "trained_vision_sha256":
            "8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e",
        "base_model": MODEL_ID,
        "base_revision": MODEL_SHA,
        "real_gpu_calls": 64,
        "paired_ood_cases": 32,
        "counts": counts,
        "peak_allocated_gib": round(peak, 3),
        "memory_cap_gib": MAX_PEAK_GIB,
        "gpu": gpu,
        "versions": versions,
        "optimizer_steps": 0,
        "owner_gold_on_gpu": False,
        "previous_consumed_dev_ood_used": False,
        "real_browser_reward_measured": False,
        "paid_gpu": False,
        "weights_published": False,
        "flagship_35b_unchanged": True,
        "predictions": predictions,
    }
    (output / "h173-paired-ood-predictions.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    return {k: v for k, v in receipt.items() if k != "predictions"}

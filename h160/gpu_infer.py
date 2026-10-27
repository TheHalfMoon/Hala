"""H160 research LoRA paired inference only, on a NEW H159 image-stable cohort.

Cross-application exploratory re-test. H158's previously consumed scenes are
never loaded, trained on, repaired or scored. Free Kaggle T4 only.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from h156.train_gpu import open_media, inference
from h158.gpu_infer import import_adapter, validated_model_action
from h160.pack_private_eval import (
    SCHEMA, SHA_H156, SHA_H157, SOURCE_ZIP,
)

SOURCE_ZIP_SHA = "95bd269c85b1ead73c01f90622ec2e943f89b50809efa1d6c53c4e33583f74d4"
MANIFEST_SHA = "c7521a6e0f359867846c15bbfabb06b1a09aa117d93d4a3b5958072d6bec92fc"
ROLES = ("before", "after")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(source):
    source = Path(source)
    if source.is_file() and digest(source.read_bytes()) != SOURCE_ZIP_SHA:
        raise ValueError("H160 frozen source package changed")
    with open_media(source) as z:
        raw = z.read("manifest.json")
        if digest(raw) != MANIFEST_SHA:
            raise ValueError("H160 frozen prompts/PNG manifest changed")
        doc = json.loads(raw)
        cases = doc.get("private_cases")
        if (doc.get("schema") != SCHEMA or len(cases) != 16 or
            doc.get("model_calls") != 0 or doc.get("optimizer_steps") != 0 or
            len({c["image_sha256"] for c in cases}) != 16 or
            len({c["index"] for c in cases}) != 16 or
            {c["index"] for c in cases} != set(range(301,317)) or
            set(z.namelist()) !=
            {"manifest.json"} | {"images/" + c["image"] for c in cases}):
            raise ValueError("H160 source cohort and model boundary violated")
        for case in cases:
            if (set(case) !=
                {"task", "language", "index", "image", "image_sha256", "instruction"}
                or not isinstance(case["instruction"], str)
                or not case["instruction"].strip()
                or case["task"] not in (
                    "complete_item", "filter_active", "clear_completed")
                or case["language"] not in ("en", "ar")
                or digest(z.read("images/" + case["image"])) !=
                   case["image_sha256"]):
                raise ValueError("H160 model source leaked gold or altered image")
    return cases


def run(source, old_zip, new_zip, output):
    import torch
    import transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from h132.grounder_contract import MODEL_ID, MODEL_SHA

    output = Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__ != "5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H160 private Kaggle FREE T4 only")
    cases = validate(source)
    old = import_adapter(old_zip, Path("/kaggle/working/h160-old-lora"), SHA_H156)
    new = import_adapter(new_zip, Path("/kaggle/working/h160-new-lora"), SHA_H157)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_SHA, trust_remote_code=False
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_SHA, torch_dtype=torch.float16,
        trust_remote_code=False, attn_implementation="sdpa"
    ).to("cuda:0")
    model = PeftModel.from_pretrained(
        base, str(old), adapter_name="prior", is_trainable=False
    )
    model.load_adapter(str(new), adapter_name="current", is_trainable=False)
    model.eval()
    results = []
    counts = Counter()
    with open_media(source) as archive:
        for role, adapter in (("before", "prior"), ("after", "current")):
            model.set_adapter(adapter)
            for case in cases:
                with Image.open(io.BytesIO(
                    archive.read("images/" + case["image"])
                )) as img:
                    screenshot = img.convert("RGB")
                if screenshot.size != (960,720):
                    raise ValueError("H160 native original screenshot size drift")
                raw = inference(processor, model, torch, screenshot, case)
                action = validated_model_action(raw)
                counts[role + "_calls"] += 1
                counts[role + "_valid"] += int(action["valid"])
                counts[role + "_click"] += int(action["click"] is not None)
                results.append({
                    "role": role, "task": case["task"],
                    "language": case["language"],
                    "image_sha": case["image_sha256"],
                    "raw": raw, "model_action": action,
                })
                if len(results) % 8 == 0:
                    print("H160_GENUINE_FREE_T4_MODEL_CALL", len(results), 32,
                          flush=True)
    if counts["before_calls"] != 16 or counts["after_calls"] != 16:
        raise ValueError("H160 incomplete real model paired inferences")
    report = {
        "schema": "hala.h160.private-stable-exploratory-gpu-inference.v1",
        "status": "H160_COMPLETED_32_REAL_GPU_CALLS",
        "source_zip_sha256": SOURCE_ZIP_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "frozen_old_lora_sha256": SHA_H156,
        "frozen_new_lora_sha256": SHA_H157,
        "gpu_calls": 32, "optimizer_steps": 0,
        "native_browser_success_measured": False,
        "h140_blind_calls": 0, "paid_gpu": False,
        "Hala35B_unchanged": True,
        "counts": dict(counts), "private_predictions": results,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "h160-private-predictions.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", "utf8"
    )
    return {k: v for k, v in report.items() if k != "private_predictions"}

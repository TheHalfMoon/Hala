"""H74: SHA-qualified pinned public Qwen model prefetch into an existing free Colab.

Designed to overlap H72's 105-part TRAIN upload. Does not train, allocate
a GPU, call a paid API, use Kaggle compute, or persist weights locally.
A metadata-only mode exists for real Colab CPU compatibility testing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
WEIGHTS_SHA = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
CACHE = Path("/tmp/hala-model-cache/hub")
RECEIPT = Path("/content/hala-h74-prefetch-receipt.json")
SCHEMA = "hala.h74.pinned-prefetch.v1"
ALLOW_PATTERNS = ["*.json", "*.txt", "model.safetensors"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def guard_free_colab() -> None:
    if not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG"):
        raise ValueError("H74 requires a real Colab runtime, not local/cloud substitution")
    if (os.environ.get("HALA_FREE_COLAB_RESEARCH") != "1"
            or os.environ.get("HALA_NO_PAID_RUNTIME_CONFIRMED") != "1"):
        raise ValueError("free non-paid Colab explicitly confirmed by operator required")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle notebook may not run H74 bootstrap")


def verify_snapshot(path: Path, *, full: bool):
    path = Path(path).resolve()
    cache_root = CACHE.resolve()
    if path == cache_root or cache_root not in path.parents:
        raise ValueError("model cache escaped the reserved Colab cache root")
    if path.name != MODEL_REVISION:
        raise ValueError("model snapshot does not match pinned revision")
    config = path / "config.json"
    if not config.is_file() or config.stat().st_size > 1024 * 1024:
        raise ValueError("missing or oversized pinned model config")
    parsed = json.loads(config.read_text(encoding="utf8"))
    if not isinstance(parsed, dict) or "model_type" not in parsed:
        raise ValueError("invalid model configuration from pinned snapshot")
    result = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
              "snapshot_config_sha256": sha256(config),
              "model_type": parsed["model_type"], "model_weights_verified": False}
    if full:
        weights = path / "model.safetensors"
        if not weights.is_file():
            raise ValueError("pinned full model weights not present")
        resolved = weights.resolve()
        if cache_root not in resolved.parents:
            raise ValueError("model weights leave pinned cache root")
        if weights.stat().st_size < 3 * 1024**3:
            raise ValueError("model weights unexpectedly short")
        if sha256(weights) != WEIGHTS_SHA:
            raise ValueError("pinned Qwen safetensors SHA256 differs")
        result["model_weights_sha256"] = WEIGHTS_SHA
        result["model_weights_bytes"] = weights.stat().st_size
        result["model_weights_verified"] = True
    return result


def write_receipt(result: dict):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    target = RECEIPT.with_name("." + RECEIPT.name + ".tmp")
    if target.exists():
        raise FileExistsError("existing partial prefetch receipt")
    with target.open("x", encoding="utf8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(target, RECEIPT)


def main() -> int:
    began = time.monotonic()
    mode = os.environ.get("HALA_H74_MODE", "")
    receipt = {"schema": SCHEMA, "status": "NOT_QUALIFIED",
               "mode": mode, "gpu_training_executed": False,
               "paid_runtime_allocated": False}
    try:
        if mode not in ("metadata", "full"):
            raise ValueError("explicit bounded metadata/full prefetch mode required")
        guard_free_colab()
        if mode == "full" and shutil.disk_usage(CACHE.parent).free < 6 * 1024**3:
            raise OSError("insufficient free Colab disk for pinned full model cache")
        from huggingface_hub import snapshot_download
        patterns = ALLOW_PATTERNS if mode == "full" else ["config.json"]
        directory = snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION,
            cache_dir=str(CACHE), allow_patterns=patterns,
            max_workers=4,
        )
        receipt.update(verify_snapshot(Path(directory), full=(mode == "full")))
        receipt["status"] = "H74_MODEL_SHA_VERIFIED" if mode == "full" else "H74_METADATA_VERIFIED"
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"[:260]
    receipt["elapsed_seconds"] = round(time.monotonic() - began, 3)
    write_receipt(receipt)
    print("H74_PREFETCH=" + json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt["status"].endswith("_VERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())

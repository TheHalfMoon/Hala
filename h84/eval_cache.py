"""H84: CPU-only, exact checkpoint-bound DEV and official eval cache admission.

Never claim a stale evaluation or silently overwrite a partial evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h60.real_drag_dev_baseline import (
    DEV_SETS, MODEL_ID, MODEL_REVISION, MAX_CASES, PROTOCOL, PROMPT_REVISION)
from h60.real_drag_dev_adapter_eval import ADAPTER_SCHEMA
from h61.official_eval_run import RUN_SCHEMA
from training.artifacts import strict_json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_receipt(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("absent or invalid bounded cached evaluation receipt")
    result = strict_json(path.read_text("utf8"))
    if not isinstance(result, dict):
        raise ValueError("invalid cached evaluation receipt")
    return result


def identity_key(kind, *, checkpoint_sha, domains, limit, max_new_tokens):
    if kind not in ("adapter", "frozen-base") or not isinstance(domains, list):
        raise ValueError("invalid evaluation run kind/domains")
    if not isinstance(checkpoint_sha, str) or (
            kind == "adapter" and (len(checkpoint_sha) != 64
                or any(c not in "0123456789abcdef" for c in checkpoint_sha))):
        raise ValueError("invalid eval model checkpoint identity")
    payload = {"kind": kind, "checkpoint_sha256": checkpoint_sha,
               "domains": domains, "limit": limit, "max_new_tokens": max_new_tokens,
               "model": MODEL_ID, "revision": MODEL_REVISION}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()


def valid_dev_cache(directory, dev, identity):
    directory = Path(directory)
    data = strict_receipt(directory / "run-result.json")
    if (dev not in DEV_SETS
            or data.get("schema") != ADAPTER_SCHEMA
            or data.get("status") != "completed"
            or data.get("dev_set") != dev
            or data.get("model_id") != MODEL_ID
            or data.get("revision") != MODEL_REVISION
            or data.get("source_sha256") != DEV_SETS[dev]["source_sha256"]
            or data.get("prompt_revision") != PROMPT_REVISION
            or data.get("coordinate_protocol") != PROTOCOL
            or data.get("split") != "public_dev"
            or data.get("evidence") != "real_inference"
            or data.get("provider") != "interactive_approved_free_colab"
            or data.get("adapters_loaded") is not True
            or data.get("model_training_executed") is not True
            or data.get("max_cases") != MAX_CASES
            or type(data.get("real_generated_cases")) is not list
            or len(data["real_generated_cases"]) != MAX_CASES
            or not isinstance(data.get("adapter_checkpoint"), dict)
            or data["adapter_checkpoint"].get("checkpoint_sha256") != identity["checkpoint_sha256"]
            or data["adapter_checkpoint"].get("adapter_weights_sha256") != identity["adapter_weights_sha256"]
            or data.get("adapter_sha256") != identity["adapter_weights_sha256"]
            or not sha_if_safe(directory / "raw-attempts.json")
            or data.get("raw_attempts_sha256") != sha_if_safe(directory / "raw-attempts.json")):
        raise ValueError("stale or unqualified cached public DEV model identity")
    return data


def sha_if_safe(path):
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > 10 * 1024 * 1024):
        return None
    return sha256(path)


def valid_official_cache(directory, *, domains, limit, kind, adapter_sha):
    directory = Path(directory)
    data = strict_receipt(directory / "official-eval-run.json")
    if (data.get("schema") != RUN_SCHEMA
            or data.get("model_id") != MODEL_ID
            or data.get("revision") != MODEL_REVISION
            or data.get("evidence") != "real_inference"
            or data.get("adapters_loaded") is not (kind == "adapter")
            or data.get("adapter_sha256") != adapter_sha
            or data.get("requested_domains") != domains
            or data.get("requested_limit") != limit
            or data.get("requested_max_new_tokens") != 160
            or not isinstance(data.get("domains"), dict)
            or set(data["domains"]) != set(domains)
            or not isinstance(data.get("raw_attempts_sha256"), dict)
            or not isinstance(data.get("score_sha256"), dict)
            or set(data["raw_attempts_sha256"]) != set(domains)
            or set(data["score_sha256"]) != set(domains)):
        raise ValueError("stale or unqualified cached official evaluation identity")
    for domain in domains:
        raw_sha = sha_if_safe(directory / domain / "raw-attempts.json")
        score_sha = sha_if_safe(directory / domain / "score.json")
        if (not raw_sha or not score_sha
                or data["raw_attempts_sha256"][domain] != raw_sha
                or data["score_sha256"][domain] != score_sha):
            raise ValueError("cached official evaluation evidence file SHA mismatch")
        if (not isinstance(data["domains"][domain], dict)
                or data["domains"][domain].get("scenes") is None):
            raise ValueError("cached official evaluation metric set invalid")
    return data["domains"]

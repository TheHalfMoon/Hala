"""H336 owner-private one-time free T4 Qwen first-click error attribution.

A prospective **source-exposed**, pretrained Qwen-only diagnostic with 12
genuine model generations and 12 independent first-party source scenes.
No training, no blind OOD, no predicted points, gold, model text, PNG,
model checkpoint or per-case reports are persisted or returned.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h336-real-qwen-first-click-source-diagnosis"
SOURCE_SHA = "ff403d13249ac0f16d806ccfe4b81c4d92df6a07"
SOURCE_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
KERNEL_FLAG = "HALA_H336_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4"
STATUS = "H336_PRIVATE_PRETRAINED_EXPOSED_FIRST_CLICK_CATEGORIES"
DIAG_STATUS = "H336_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE"
REPORT = "hala-h336-private-first-click-categories.json"
DIAG = "hala-h336-private-first-click-stage.json"
STAGES = frozenset({"ENV", "SOURCE", "DEPENDENCIES",
                    "MODEL_LOAD", "REAL_BROWSER", "PRIVATE_AGGREGATE"})
EXCEPTIONS = frozenset({"ValueError", "RuntimeError", "OSError", "ImportError",
                        "ModuleNotFoundError", "CalledProcessError",
                        "FileNotFoundError", "OutOfMemoryError", "TimeoutError"})
STAGE = "ENV"


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H336 PRIVATE_PRETRAINED_DIAGNOSTIC_REFUSED")


def verify_free(torch) -> dict:
    working = Path("/kaggle/working")
    need(sys.platform == "linux" and working.is_dir()
         and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
         and os.environ.get(KERNEL_FLAG) == "1"
         and torch.cuda.is_available()
         and torch.cuda.device_count() in (1, 2)
         and all("T4" in torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count()))
         and shutil.disk_usage("/tmp").free >= 8 * 1024**3
         and not (working / REPORT).exists()
         and not (working / DIAG).exists())
    return {"free_owner_private_t4": True,
            "t4_count": torch.cuda.device_count()}


def clone_pinned_source(dest: Path) -> Path:
    need(not dest.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone",
         "--depth", "20", "--single-branch", "--branch",
         SOURCE_BRANCH, SOURCE_REPO, str(dest)],
        check=True, capture_output=True, text=True, timeout=180)
    # The source is frozen at the earlier H336 model-boundary commit,
    # not the moving HEAD of this code-only kernel builder branch.
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--detach", SOURCE_SHA],
        check=True, capture_output=True, text=True, timeout=30)
    actual = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.strip()
    need(actual == SOURCE_SHA and dirty == "")
    html = dest / "h319" / "beaconboard.html"
    need(html.is_file() and not html.is_symlink()
         and not html.parent.is_symlink())
    raw = html.read_bytes()
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == SOURCE_BLOB)
    return html


def validate_aggregate(result: object) -> dict:
    """Exact low-entropy source-only counts, never a model identity proof."""
    from h335.original_first_click_error_attribution import CATEGORIES
    from h336.preloaded_first_click_study import ROLE
    need(type(result) is dict and set(result) == {
        "status", "frozen_exposed_scenes",
        "claimed_preloaded_model_generations", "claimed_native_clicks",
        "by_language", "real_model_identity_requires_external_attestation",
        "official_train_admitted", "fresh_blind_ood",
        "learned_adapter_gain", "can_release"})
    for key, expected in (
        ("status", ROLE), ("frozen_exposed_scenes", 12),
        ("claimed_preloaded_model_generations", 12),
        ("real_model_identity_requires_external_attestation", True),
        ("official_train_admitted", False), ("fresh_blind_ood", False),
        ("learned_adapter_gain", False), ("can_release", False),
    ):
        need(type(result[key]) is type(expected) and result[key] == expected)
    native_clicks = result["claimed_native_clicks"]
    need(type(native_clicks) is int and 0 <= native_clicks <= 12)
    groups = result["by_language"]
    need(type(groups) is dict and set(groups) == {"en", "ar"})
    counted = 0
    for language in ("en", "ar"):
        group = groups[language]
        need(type(group) is dict and set(group) == {
            "scenes", "native_actions", "categories"})
        categories = group["categories"]
        need(type(categories) is dict and set(categories) == set(CATEGORIES)
             and all(type(n) is int and 0 <= n <= 6
                     for n in categories.values())
             and sum(categories.values()) == 6
             and type(group["scenes"]) is int
             and group["scenes"] == 6
             and type(group["native_actions"]) is int
             and group["native_actions"] == 6 - categories["abstain"])
        counted += group["native_actions"]
    need(counted == native_clicks)
    return {"by_language": groups, "first_click_model_generations": 12,
            "public_source_exposed_not_blind": True,
            "adapted_gain_proven": False, "can_release": False}


def safe_report(aggregate: object, hardware: object) -> dict:
    validate_aggregate(aggregate)
    need(type(hardware) is dict and set(hardware) == {
        "free_owner_private_t4", "t4_count"}
         and hardware["free_owner_private_t4"] is True
         and type(hardware["t4_count"]) is int
         and hardware["t4_count"] in (1, 2))
    return {
        "status": STATUS,
        "source_commit": SOURCE_SHA,
        "source_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "hardware": hardware, "source_exposed_first_click_counts": aggregate,
        "new_optimizer_steps": 0,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "trained_model_gain_proven": False, "can_release": False,
        "raw_private_model_actions_media_gold_coordinates_saved": False,
    }


def safe_failure_receipt(stage: str, name: str) -> dict:
    need(type(stage) is str and stage in STAGES and type(name) is str)
    return {
        "status": DIAG_STATUS,
        "failure_stage": stage,
        "exception_class": name if name in EXCEPTIONS
                           else "OTHER_ERROR_CLASS",
        "source_commit": SOURCE_SHA,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "model_skill_verified": False, "blind_ood_proven": False,
        "adapted_gain_proven": False,
        "private_media_or_raw_exception_saved": False,
    }


def write_failure(stage: str, error: BaseException) -> None:
    working = Path("/kaggle/working")
    if (sys.platform != "linux" or not working.is_dir()
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get(KERNEL_FLAG) != "1"):
        return
    target = working / DIAG
    if not target.exists():
        target.write_text(json.dumps(
            safe_failure_receipt(stage, type(error).__name__),
            sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")


def run() -> None:
    global STAGE
    import torch
    hardware = verify_free(torch)
    root = Path("/tmp/hala-h336-first-click-source-model")
    need(not root.exists())
    root.mkdir(mode=0o700)
    try:
        STAGE = "SOURCE"
        html = clone_pinned_source(root / "source")
        sys.path.insert(0, str(html.parents[1]))
        STAGE = "DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import (
            verify_exact_lmfe_installed_source)
        from h336.preloaded_first_click_study import (
            ROLE, run_preloaded_first_click_study)
        from h319.beaconboard_dynamic_research import checked_source
        need(transformers.__version__ == "5.16.1")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update({
            "HF_HOME": str(root / "hf"),
            "HF_HUB_CACHE": str(root / "hf" / "hub"),
            "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        })
        STAGE = "MODEL_LOAD"
        torch.manual_seed(336)
        cache = str(root / "hf" / "hub")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        need(getattr(model.config, "model_type", None) == "qwen3_vl")
        STAGE = "REAL_BROWSER"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True, args=["--no-sandbox"])
            try:
                aggregate = run_preloaded_first_click_study(
                    browser, html, processor, model, role=ROLE)
            finally:
                browser.close()
        STAGE = "PRIVATE_AGGREGATE"
        report = safe_report(aggregate, hardware)
        serial = json.dumps(report, sort_keys=True, separators=(",", ":"))
        need(len(serial.encode("utf-8")) <= 4096)
        (Path("/kaggle/working") / REPORT).write_text(
            serial + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as err:
        write_failure(STAGE, err)
        raise RuntimeError(
            "H336 private first-click research stopped. "
            "Inspect ONLY strict categorical allowlisted report."
        ) from None

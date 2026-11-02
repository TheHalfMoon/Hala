"""H351: owner-private FREE T4 prospective REAL pretrained Qwen miss-distance experiment.

PUBLIC, already-exposed original source only. No new training or OOD.
Never emit raw Qwen text, action points, screenshots, labels, source targets,
HTML, per-scene failures, model weights, or private runtime exceptions.
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

REPO = "https://github.com/TheHalfMoon/Hala.git"
BRANCH = "ps/h350-development-training-official-admission"
PIN = "b925b2d9c66ff2a8a38e766eb1eec03d09c4d61c"
HTML_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
FLAG = "HALA_H351_PRIVATE_NONCOMMERCIAL_FREE_T4"
REPORT = "hala-h351-exposed-qwen-miss-bands.json"
STATUS = "H351_PRIVATE_PRETRAINED_EXPOSED_FIRST_CLICK_MISS_BANDS"
STAGES = frozenset(("ENV", "SOURCE", "DEPENDENCIES", "MODEL_LOAD",
                    "BROWSER", "REPORT"))
STAGE = "ENV"
ERROR_FILE = "hala-h351-private-stage-only.json"
ALLOWED_ERRORS = frozenset(("ValueError", "RuntimeError", "OSError",
                           "ImportError", "ModuleNotFoundError",
                           "CalledProcessError", "FileNotFoundError",
                           "OutOfMemoryError", "TimeoutError"))

def need(ok):
    if not ok:
        raise ValueError("H351 PRIVATE PRETRAINED MISS DIAGNOSTIC REFUSED")

def qualify_free(torch):
    working = Path("/kaggle/working")
    need(sys.platform == "linux" and working.is_dir()
         and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
         and os.environ.get(FLAG) == "1"
         and torch.cuda.is_available()
         and torch.cuda.device_count() in (1, 2)
         and all("T4" in torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count()))
         and shutil.disk_usage("/tmp").free >= 8 * 1024**3
         and not (working / REPORT).exists()
         and not (working / ERROR_FILE).exists())
    return {"free_owner_private_t4": True,
            "t4_count": torch.cuda.device_count()}

def original_pinned_source(destination):
    need(not destination.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
         "--single-branch", "--branch", BRANCH, REPO, str(destination)],
        check=True, capture_output=True, text=True, timeout=180)
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain"],
        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    need(head == PIN and dirty == "")
    html = destination / "h319" / "beaconboard.html"
    need(html.is_file() and not html.is_symlink()
         and not html.parent.is_symlink())
    raw = html.read_bytes()
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == HTML_BLOB)
    return html

def validate_observed_counts(source_counts, summaries):
    """Accept source-native categories only with 12 observed wrapped generations.

    The H337 source helper intentionally marks its own model count ZERO;
    this separate experimental layer checks each real policy generation.
    """
    from h337.source_miss_distance_bands import CATEGORIES, ROLE
    need(type(source_counts) is dict and set(source_counts) == {
        "status", "source_public_exposed_scenes", "by_language",
        "source_only_not_model_attested", "new_model_generations",
        "new_optimizer_steps", "official_train_admitted", "fresh_blind_ood",
        "learned_gain_proven", "can_release"})
    need(source_counts["status"] == ROLE
         and source_counts["source_public_exposed_scenes"] == 12
         and source_counts["source_only_not_model_attested"] is True
         and all(source_counts[k] is False for k in (
             "official_train_admitted", "fresh_blind_ood",
             "learned_gain_proven", "can_release"))
         and source_counts["new_model_generations"] == 0
         and source_counts["new_optimizer_steps"] == 0)
    groups = source_counts["by_language"]
    need(type(groups) is dict and set(groups) == {"en", "ar"})
    for language in ("en", "ar"):
        group = groups[language]
        need(type(group) is dict
             and set(group) == {"scenes", "native_actions", "categories"}
             and type(group["scenes"]) is int and group["scenes"] == 6)
        cats = group["categories"]
        need(type(cats) is dict and set(cats) == set(CATEGORIES)
             and all(type(n) is int and 0 <= n <= 6
                     for n in cats.values()) and sum(cats.values()) == 6
             and type(group["native_actions"]) is int
             and group["native_actions"] == 6 - cats["abstain"])
    need(type(summaries) is list and len(summaries) == 12)
    for summary in summaries:
        need(type(summary) is dict
             and summary.get("model_generations_via_wrapped_generate") == 1
             and type(summary.get("model_generations_via_wrapped_generate")) is int
             and summary.get("model_time_token_filters_installed") == 1
             and type(summary.get("model_time_token_filters_installed")) is int
             and summary.get("distinct_original_pngs_observed") == 1
             and type(summary.get("distinct_original_pngs_observed")) is int
             and summary.get("type_calls") == 0
             and summary.get("pretrained_model_identity_attested") is False
             and summary.get("real_owner_private_gpu_run_attested") is False
             and summary.get("generalizable_hala_gain_proven") is False)
    return groups

def safe_report(source_counts, summaries, hardware):
    groups = validate_observed_counts(source_counts, summaries)
    need(type(hardware) is dict
         and set(hardware) == {"free_owner_private_t4", "t4_count"}
         and hardware["free_owner_private_t4"] is True
         and type(hardware["t4_count"]) is int
         and hardware["t4_count"] in (1, 2))
    return {
        "status": STATUS, "source_commit": PIN, "source_html_blob": HTML_BLOB,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "hardware": hardware, "by_language": groups,
        "claimed_wrapped_model_generations": 12,
        "original_source_public_exposed": True,
        "model_identity_requires_separate_job_attestation": True,
        "official_train_admitted": False, "fresh_blind_ood": False,
        "adapted_model_gain_proven": False, "new_optimizer_steps": 0,
        "can_release": False, "raw_predictions_or_media_exported": False,
    }

def run():
    global STAGE
    import torch
    hw = qualify_free(torch)
    scratch = Path("/tmp/hala-h351-qwen-miss-proximity")
    need(not scratch.exists())
    scratch.mkdir(mode=0o700)
    try:
        STAGE = "SOURCE"
        html = original_pinned_source(scratch / "source")
        sys.path.insert(0, str(html.parents[1]))
        STAGE = "DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import (
            verify_exact_lmfe_installed_source)
        from h323.preloaded_dynamic_visual_policy import PreloadedDynamicVisualPolicy
        from h319.beaconboard_dynamic_research import exact_task, checked_source
        from h337.source_miss_distance_bands import audit_source_distance, ROLE
        need(transformers.__version__ == "5.16.1")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update(HF_HOME=str(scratch / "hf"),
                          HF_HUB_CACHE=str(scratch / "hf" / "hub"),
                          HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        STAGE = "MODEL_LOAD"
        torch.manual_seed(351)
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV,
            cache_dir=str(scratch / "hf" / "hub"), trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REV,
            cache_dir=str(scratch / "hf" / "hub"),
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        need(getattr(model.config, "model_type", None) == "qwen3_vl")
        policies = []
        def factory(scene):
            policy = PreloadedDynamicVisualPolicy(
                processor, model, exact_task(scene))
            policies.append(policy)
            return policy
        STAGE = "BROWSER"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True, args=["--no-sandbox"])
            try:
                counts = audit_source_distance(
                    browser, html, factory, role=ROLE)
            finally:
                browser.close()
        summaries = [p.safe_contract_summary() for p in policies]
        STAGE = "REPORT"
        report = safe_report(counts, summaries, hw)
        raw = json.dumps(report, sort_keys=True, separators=(",", ":"))
        need(len(raw.encode("utf-8")) <= 4096)
        (Path("/kaggle/working") / REPORT).write_text(
            raw + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

def stage_only_receipt():
    return {
        "status": "H351_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE",
        "stage": STAGE if STAGE in STAGES else "ENV",
        "official_train_admitted": False, "adapted_gain_proven": False,
        "fresh_blind_ood": False, "can_release": False,
        "raw_private_model_data_saved": False,
    }

if __name__ == "__main__":
    try:
        run()
    except Exception:
        if (sys.platform == "linux" and Path("/kaggle/working").is_dir()
                and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                and os.environ.get(FLAG) == "1"
                and not (Path("/kaggle/working") / ERROR_FILE).exists()):
            (Path("/kaggle/working") / ERROR_FILE).write_text(
                json.dumps(stage_only_receipt(), sort_keys=True) + "\n",
                encoding="utf-8")
        raise RuntimeError("H351 private diagnostic failed; only stage is available") from None

"""H333: ONE prospective owner-private FREE-T4 matched public/exposed prompt run.

This is pretrained Qwen research on source-exposed original BeaconBoard.
No optimizer/training, no blind OOD, no adapted-gain claim, no private media.
No GPU run has been submitted or completed by merely committing this file.
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
SOURCE_BRANCH = "ps/h332-paired-exposed-prompt-research-protocol"
SOURCE_SHA = "087ef18b7fc5c71047ad6e5207d8d0530380c4c1"
SOURCE_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
KERNEL_FLAG = "HALA_H333_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4"
REPORT = "hala-h333-private-paired-prompt-exposed-aggregate.json"
DIAG = "hala-h333-private-paired-prompt-stage-diagnostic.json"
STATUS = "H333_PRETRAINED_PUBLIC_EXPOSED_MATCHED_PROMPT_RESEARCH_ONLY"
DIAG_STATUS = "H333_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE"
STAGES = frozenset({"ENV", "SOURCE", "DEPENDENCIES", "MODEL_LOAD",
                    "REAL_BROWSER", "PRIVATE_AGGREGATE"})
EXCEPTIONS = frozenset({
    "ValueError", "RuntimeError", "OSError", "ImportError",
    "ModuleNotFoundError", "CalledProcessError", "FileNotFoundError",
    "OutOfMemoryError", "TimeoutError",
})
STAGE = "ENV"


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H333 EXPOSED_PRIVATE_MATCHED_STUDY_REFUSED")


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


def clone_source(dest: Path) -> Path:
    need(not dest.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
         "--single-branch", "--branch", SOURCE_BRANCH, SOURCE_REPO, str(dest)],
        check=True, capture_output=True, text=True)
    sha = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.strip()
    need(sha == SOURCE_SHA and not dirty)
    html = dest / "h319" / "beaconboard.html"
    need(html.is_file() and not html.is_symlink()
         and not html.parent.is_symlink())
    raw = html.read_bytes()
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == SOURCE_BLOB)
    return html


def safe_failure_receipt(stage: str, kind: str) -> dict:
    need(type(stage) is str and stage in STAGES
         and type(kind) is str)
    return {
        "status": DIAG_STATUS,
        "failure_stage": stage,
        "exception_class": kind if kind in EXCEPTIONS
                           else "OTHER_ERROR_CLASS",
        "source_commit": SOURCE_SHA,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "model_skill_verified": False,
        "blind_ood_proven": False,
        "adapted_gain_proven": False,
        "private_media_or_raw_exception_saved": False,
    }


def write_private_failure(stage: str, err: BaseException) -> None:
    working = Path("/kaggle/working")
    if (sys.platform != "linux" or not working.is_dir()
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get(KERNEL_FLAG) != "1"):
        return
    dest = working / DIAG
    if not dest.exists():
        dest.write_text(json.dumps(safe_failure_receipt(
            stage, type(err).__name__), separators=(",", ":"),
            sort_keys=True) + "\n", encoding="utf-8")


def validate_aggregate(result: object) -> dict:
    """Verify fixed arithmetic of low-entropy paired source-only summary."""
    from h332.paired_exposed_prompt_research import ROLE, OUTCOMES
    need(type(result) is dict and set(result) == {
        "status", "frozen_public_scene_count", "matched_model_source_runs",
        "english_scene_pairs", "arabic_rtl_scene_pairs",
        "counterbalanced_orders", "by_language",
        "source_public_exposed_not_blind",
        "genuine_model_identity_requires_external_attestation",
        "official_train_dev_ood_admitted", "optimizer_steps",
        "adapted_model_gain_proven", "fresh_blind_ood_proven", "can_release",
    })
    for key, value in (
        ("status", ROLE), ("frozen_public_scene_count", 4),
        ("matched_model_source_runs", 8), ("english_scene_pairs", 2),
        ("arabic_rtl_scene_pairs", 2), ("counterbalanced_orders", True),
        ("source_public_exposed_not_blind", True),
        ("genuine_model_identity_requires_external_attestation", True),
        ("official_train_dev_ood_admitted", False), ("optimizer_steps", 0),
        ("adapted_model_gain_proven", False), ("fresh_blind_ood_proven", False),
        ("can_release", False),
    ):
        need(type(result[key]) is type(value) and result[key] == value)
    groups = result["by_language"]
    need(type(groups) is dict and set(groups) == {"en", "ar"})
    for language in ("en", "ar"):
        group = groups[language]
        need(type(group) is dict and set(group) == {
            "control", "candidate", "pair_outcomes"})
        wins = group["pair_outcomes"]
        need(type(wins) is dict and set(wins) == {"control", "candidate", "tie"}
             and all(type(n) is int and 0 <= n <= 2 for n in wins.values())
             and sum(wins.values()) == 2)
        for arm in ("control", "candidate"):
            item = group[arm]
            need(type(item) is dict and set(item) == {
                "tasks", "task_success", "model_generations",
                "native_actions", "failure_counts"})
            failure = item["failure_counts"]
            need(type(failure) is dict and set(failure) == set(OUTCOMES)
                 and all(type(n) is int and 0 <= n <= 2
                         for n in failure.values())
                 and sum(failure.values()) == 2)
            need(type(item["tasks"]) is int and item["tasks"] == 2
                 and type(item["task_success"]) is int
                 and item["task_success"] == failure["none"])
            min_gen = (2*failure["none"] + failure["abstain"]
                       + failure["wrong_focus_or_state"]
                       + 2*failure["wrong_code"]
                       + 2*failure["source_state_mismatch"])
            min_dispatch = (2*failure["none"]
                            + failure["wrong_focus_or_state"]
                            + failure["wrong_code"]
                            + 2*failure["source_state_mismatch"])
            generations, actions = item["model_generations"], item["native_actions"]
            need(type(generations) is int and type(actions) is int
                 and 2 <= generations <= 4 and 0 <= actions <= 4
                 and 0 <= generations - min_gen <= failure["abstain"]
                 and generations - min_gen == actions - min_dispatch)
        need(group["candidate"]["task_success"]
             - group["control"]["task_success"]
             == wins["candidate"] - wins["control"])
    return {"matched_model_source_runs": 8,
            "public_exposed_not_blind": True, "adapted_gain_proven": False}


def safe_report(source_aggregate: object, hardware: object) -> dict:
    # Source-side real evaluator owns aggregation; independent verifier
    # must check exact external private job/code and internal schema.
    from h332.paired_exposed_prompt_research import ROLE, OUTCOMES
    need(type(source_aggregate) is dict
         and source_aggregate.get("status") == ROLE
         and source_aggregate.get("frozen_public_scene_count") == 4
         and source_aggregate.get("matched_model_source_runs") == 8
         and source_aggregate.get("genuine_model_identity_requires_external_attestation") is True
         and source_aggregate.get("source_public_exposed_not_blind") is True
         and source_aggregate.get("official_train_dev_ood_admitted") is False
         and source_aggregate.get("fresh_blind_ood_proven") is False
         and source_aggregate.get("adapted_model_gain_proven") is False
         and source_aggregate.get("can_release") is False
         and source_aggregate.get("optimizer_steps") == 0
         and type(hardware) is dict
         and hardware.get("free_owner_private_t4") is True
         and type(hardware.get("t4_count")) is int
         and hardware["t4_count"] in (1, 2))
    verified = validate_aggregate(source_aggregate)
    need(verified["matched_model_source_runs"] == 8)
    return {
        "status": STATUS,
        "source_commit": SOURCE_SHA,
        "source_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": hardware,
        "exposed_paired_source_aggregate": source_aggregate,
        "new_optimizer_steps": 0,
        "official_train_dev_ood_admitted": False,
        "independent_blind_ood": False,
        "trained_model_gain_proven": False,
        "private_media_model_text_gold_coordinates_or_weights_saved": False,
    }


def run() -> None:
    global STAGE
    import torch
    hardware = verify_free(torch)
    root = Path("/tmp/hala-h333-private-paired-research")
    need(not root.exists())
    root.mkdir(mode=0o700)
    try:
        STAGE = "SOURCE"
        html = clone_source(root / "source")
        sys.path.insert(0, str(html.parents[1]))
        STAGE = "DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import (
            verify_exact_lmfe_installed_source,
        )
        from h319.beaconboard_dynamic_research import checked_source
        from h332.paired_exposed_prompt_research import run_paired_exposed_study, ROLE
        need(transformers.__version__ == "5.16.1")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update(
            HF_HOME=str(root / "hf"),
            HF_HUB_CACHE=str(root / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
        )
        STAGE = "MODEL_LOAD"
        torch.manual_seed(333)
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
                aggregate = run_paired_exposed_study(
                    browser, html, processor, model, role=ROLE)
            finally:
                browser.close()
        STAGE = "PRIVATE_AGGREGATE"
        receipt = safe_report(aggregate, hardware)
        serialized = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
        need(len(serialized.encode("utf-8")) <= 4096)
        (Path("/kaggle/working") / REPORT).write_text(
            serialized + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as err:
        write_private_failure(STAGE, err)
        raise RuntimeError(
            "H333 owner-private paired research stopped; "
            "inspect ONLY the strict categorical allowlisted report"
        ) from None

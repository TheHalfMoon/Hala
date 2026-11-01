"""H324: prospective PRIVATE owner FREE T4 pretrained BeaconBoard pilot.

Executable only after external owner-private/noncommercial/free-quota preflight.
Original public/exposed source research; never blind OOD or adapted gain.
Never save or print model text, screenshots, action points, task/gold, weights.
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
BRANCH = "ps/h323-preloaded-qwen-dynamic-beaconboard-bridge"
HEAD = "651364ab952c7e90dabb4c7c240e312ebd8d417f"
HTML_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
FLAG = "HALA_H324_OWNER_PRIVATE_NONCOMMERCIAL_FREE_T4"
RESULT = "hala-h324-private-beaconboard-aggregate.json"
DIAGNOSTIC = "hala-h324-private-beaconboard-stage-only.json"
STATUS = "H324_REAL_PRETRAINED_PUBLIC_EXPOSED_BEACONBOARD_NOT_OOD"
STAGES = frozenset(("ENV", "SOURCE", "DEPENDENCIES", "MODEL_LOAD",
                    "REAL_BROWSER", "AGGREGATE", "CLEANUP"))
INDICES = (0, 6)  # prospectively frozen EN/AR pair, not chosen post-result
_ALLOWED_FAILURES = frozenset(("none", "abstain", "wrong_focus_or_state",
                               "wrong_code", "source_state_mismatch"))
ROW_KEYS = frozenset((
    "status", "source_original_task_success",
    "real_model_generations_claimed_by_wrapper", "native_actions_dispatched",
    "source_categorical_failure", "pretrained_model_identity_independently_attested",
    "source_public_exposed", "official_train_dev_ood_admitted",
    "blind_ood_proven", "trained_model_gain_proven", "can_release",
))
_STAGE = "ENV"


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H324 PRIVATE STUDY CONTRACT REFUSED")


def attest_free_runtime(torch: object) -> dict:
    """NOT a substitute for external owner Kaggle account/privacy/quota check."""
    working = Path("/kaggle/working")
    require(sys.platform == "linux" and working.is_dir()
            and bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
            and os.environ.get(FLAG) == "1"
            and callable(getattr(torch.cuda, "is_available", None))
            and torch.cuda.is_available()
            and torch.cuda.device_count() in (1, 2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count()))
            and shutil.disk_usage("/tmp").free >= 8 * 1024**3
            and not (working / RESULT).exists()
            and not (working / DIAGNOSTIC).exists())
    return {"free_t4": True, "t4_count": torch.cuda.device_count()}


def aggregate(rows: object, hardware: object) -> dict:
    """Aggregate ONLY two real source outcomes; reject fabricated/extra fields.

    Does not itself attest model/GPU: caller must verify those independently.
    """
    require(type(hardware) is dict and set(hardware) == {"free_t4", "t4_count"}
            and hardware["free_t4"] is True
            and type(hardware["t4_count"]) is int
            and hardware["t4_count"] in (1, 2))
    require(type(rows) is list and len(rows) == len(INDICES))
    generations = dispatches = successes = abstentions = 0
    for row in rows:
        require(type(row) is dict and set(row) == ROW_KEYS)
        require(row["status"] ==
                "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY"
                and row["source_public_exposed"] is True
                and all(row[key] is False for key in (
                    "pretrained_model_identity_independently_attested",
                    "official_train_dev_ood_admitted", "blind_ood_proven",
                    "trained_model_gain_proven", "can_release"))
                and type(row["source_original_task_success"]) is bool
                and type(row["real_model_generations_claimed_by_wrapper"]) is int
                and type(row["native_actions_dispatched"]) is int
                and type(row["source_categorical_failure"]) is str
                and row["source_categorical_failure"] in _ALLOWED_FAILURES)
        n = row["real_model_generations_claimed_by_wrapper"]
        d = row["native_actions_dispatched"]
        ok = row["source_original_task_success"]
        why = row["source_categorical_failure"]
        require((n, d, ok, why) in (
            (1, 0, False, "abstain"),
            (1, 1, False, "wrong_focus_or_state"),
            (2, 1, False, "abstain"),
            (2, 1, False, "wrong_code"),
            (2, 2, False, "source_state_mismatch"),
            (2, 2, True, "none"),
        ))
        generations += n
        dispatches += d
        successes += int(ok)
        abstentions += int(why == "abstain")
    return {
        "status": STATUS,
        "source_commit": HEAD, "original_html_git_blob": HTML_BLOB,
        "model_id": MODEL, "model_revision": REVISION,
        "hardware": hardware,
        "source_research_scenes": 2, "languages_frozen": "one_en_one_ar",
        "real_pretrained_model_generations": generations,
        "native_browser_actions_dispatched": dispatches,
        "original_four_station_task_successes": successes,
        "source_abstentions": abstentions,
        "optimizer_steps": 0,
        "public_exposed_not_blind_ood": True,
        "official_train_dev_ood_admitted": False,
        "adapted_gain_proven": False, "can_release": False,
        "raw_model_text_images_points_gold_weights_saved": False,
    }


def safe_failure(stage: str, kind: str) -> dict:
    require(stage in STAGES and type(kind) is str)
    allowed = {"ValueError", "RuntimeError", "OSError", "ImportError",
               "ModuleNotFoundError", "CalledProcessError",
               "FileNotFoundError", "OutOfMemoryError"}
    return {
        "status": "H324_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE",
        "stage": stage,
        "exception_class": kind if kind in allowed else "OTHER",
        "source_commit": HEAD,
        "model_revision": REVISION,
        "verified_score": False, "optimizer_steps": 0,
        "raw_media_predictions_gold_saved": False,
    }


def write_create_only(path: Path, document: dict) -> None:
    """No overwrite, no console logging, owner-private output only."""
    require(path.parent == Path("/kaggle/working")
            and path.name in (RESULT, DIAGNOSTIC))
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def checked_checkout(root: Path) -> Path:
    require(not root.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
         "--single-branch", "--branch", BRANCH, REPO, str(root)],
        capture_output=True, text=True, check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    require(head == HEAD and not dirty)
    html = root / "h319" / "beaconboard.html"
    require(html.is_file() and not html.is_symlink())
    raw = html.read_bytes()
    digest = sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()
    require(digest == HTML_BLOB)
    return html


def run() -> None:
    """Never run on Windows; do not launch without verified free private quota."""
    global _STAGE
    import torch
    hardware = attest_free_runtime(torch)
    root = Path("/tmp/hala-h324-private-beaconboard-pretrained")
    require(not root.exists())
    root.mkdir(mode=0o700)
    report = None
    try:
        _STAGE = "SOURCE"
        html = checked_checkout(root / "source")
        sys.path.insert(0, str(html.parents[1]))
        _STAGE = "DEPENDENCIES"
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import verify_exact_lmfe_installed_source
        from h319.beaconboard_dynamic_research import SCENES, checked_source
        from h323.preloaded_dynamic_bridge import run_preloaded_dynamic_source_research
        require(transformers.__version__ == "5.16.1")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update(
            HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        _STAGE = "MODEL_LOAD"
        torch.manual_seed(324)
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, cache_dir=str(root / "hf" / "hub"),
            trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, cache_dir=str(root / "hf" / "hub"),
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(getattr(model.config, "model_type", None) == "qwen3_vl")
        _STAGE = "REAL_BROWSER"
        rows = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                for index in INDICES:
                    rows.append(run_preloaded_dynamic_source_research(
                        browser, html, processor, model, SCENES[index]))
            finally:
                browser.close()
        _STAGE = "AGGREGATE"
        report = aggregate(rows, hardware)
        del rows, model, processor
    finally:
        _STAGE = "CLEANUP"
        shutil.rmtree(root)
        require(not root.exists())
    require(report is not None)
    write_create_only(Path("/kaggle/working") / RESULT, report)


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        if (sys.platform == "linux"
                and Path("/kaggle/working").is_dir()
                and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                and os.environ.get(FLAG) == "1"):
            try:
                write_create_only(
                    Path("/kaggle/working") / DIAGNOSTIC,
                    safe_failure(_STAGE, type(error).__name__))
            except (OSError, ValueError):
                pass
        raise RuntimeError("H324 PRIVATE PILOT REFUSED; NO SCORE CLAIM") from None

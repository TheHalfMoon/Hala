"""H324: owner-private FREE noncommercial T4 genuine pretrained BeaconBoard pilot.

Two frozen PUBLIC/EXPOSED original-source scenes, one EN and one Arabic RTL.
Never writes or prints model outputs, screenshots, gold, action coordinates,
private artifacts or weights. Reports only bounded categorical outcomes.
NO training, adapted model, blind OOD, or legal data admission.
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
SOURCE_BRANCH = "ps/h323-preloaded-qwen-dynamic-beaconboard-bridge"
SOURCE_SHA = "651364ab952c7e90dabb4c7c240e312ebd8d417f"
SOURCE_BLOB = "6e8467e123c7f0fbac77ea553622ab8e659f9c16"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
KERNEL_FLAG = "HALA_H324_PRIVATE_FREE_NONCOMMERCIAL_T4"
REPORT = "hala-h324-beaconboard-private-two-case-aggregate.json"
DIAG = "hala-h324-beaconboard-private-stage-diagnostic.json"
STATUS = "H324_PRETRAINED_ORIGINAL_BEACONBOARD_PUBLIC_EXPOSED_RESEARCH"
DIAG_STATUS = "H324_PRIVATE_STAGE_ONLY_NOT_MODEL_SKILL"
STAGES = frozenset({"ENV", "SOURCE", "DEPENDENCIES", "MODEL_LOAD",
                    "REAL_BROWSER", "PRIVATE_AGGREGATE"})
FAILURES = frozenset({
    "none", "abstain", "wrong_focus_or_state",
    "wrong_code", "source_state_mismatch",
})
EXCEPTIONS = frozenset({
    "ValueError", "RuntimeError", "OSError", "ImportError",
    "ModuleNotFoundError", "CalledProcessError", "FileNotFoundError",
    "OutOfMemoryError",
})
SCENE_INDICES = (0, 6)
STAGE = "ENV"


def need(ok: bool) -> None:
    if not ok:
        raise ValueError("H324 PRIVATE_PRETRAINED_DYNAMIC_REFUSED")


def verify_free(torch) -> dict:
    directory = Path("/kaggle/working")
    need(sys.platform == "linux" and directory.is_dir()
         and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
         and os.environ.get(KERNEL_FLAG) == "1"
         and torch.cuda.is_available()
         and torch.cuda.device_count() in (1, 2)
         and all("T4" in torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count()))
         and shutil.disk_usage("/tmp").free >= 8 * 1024**3
         and not (directory / REPORT).exists()
         and not (directory / DIAG).exists())
    return {"free_t4": True, "devices": torch.cuda.device_count()}


def clone_source(dest: Path) -> Path:
    need(not dest.exists())
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
         "--single-branch", "--branch", SOURCE_BRANCH, SOURCE_REPO, str(dest)],
        check=True, capture_output=True, text=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(dest), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.strip()
    need(head == SOURCE_SHA and not dirty)
    html = dest / "h319" / "beaconboard.html"
    need(html.is_file() and not html.is_symlink()
         and not html.parent.is_symlink())
    raw = html.read_bytes()
    blob = sha1(b"blob " + str(len(raw)).encode("ascii")
                + b"\x00" + raw).hexdigest()
    need(blob == SOURCE_BLOB)
    return html


def strict_row(row: object, language: str) -> dict:
    """Only source-scored boolean/categorical actions, never source details."""
    need(type(row) is dict and set(row) == {
        "language", "model_generations", "native_dispatched",
        "source_failure", "task_success",
    })
    need(row["language"] == language
         and type(row["model_generations"]) is int
         and row["model_generations"] in (1, 2)
         and type(row["native_dispatched"]) is int
         and row["native_dispatched"] in (0, 1, 2)
         and type(row["source_failure"]) is str
         and row["source_failure"] in FAILURES
         and type(row["task_success"]) is bool)
    n, dispatched, failure = (row["model_generations"],
                              row["native_dispatched"],
                              row["source_failure"])
    need(
        (n == 1 and dispatched == 0 and failure == "abstain")
        or (n == 1 and dispatched == 1 and failure == "wrong_focus_or_state")
        or (n == 2 and dispatched == 1 and failure in {"abstain", "wrong_code"})
        or (n == 2 and dispatched == 2
            and failure in {"none", "source_state_mismatch"})
    )
    need(row["task_success"] is (failure == "none"))
    return dict(row)


def verified_report(rows: object, hardware: object) -> dict:
    need(type(hardware) is dict and set(hardware) == {"free_t4", "devices"}
         and hardware["free_t4"] is True
         and type(hardware["devices"]) is int
         and hardware["devices"] in (1, 2))
    need(type(rows) is list and len(rows) == 2)
    safe = [strict_row(row, language) for row, language in
            zip(rows, ("en", "ar"), strict=True)]
    return {
        "status": STATUS,
        "source_commit": SOURCE_SHA,
        "source_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": hardware,
        "research_cases": 2,
        "model_generations": sum(row["model_generations"] for row in safe),
        "native_actions_dispatched":
            sum(row["native_dispatched"] for row in safe),
        "task_successes": sum(int(row["task_success"]) for row in safe),
        "rows": safe,
        "public_exposed_not_blind": True,
        "pretrained_optimizer_steps": 0,
        "official_train_dev_ood_admitted": False,
        "independent_blind_ood": False,
        "adapted_model_gain_proven": False,
        "raw_media_gold_model_output_coordinates_or_weights_saved": False,
    }


def safe_failure_receipt(stage: str, kind: str) -> dict:
    need(stage in STAGES and type(kind) is str)
    return {
        "status": DIAG_STATUS,
        "failure_stage": stage,
        "exception_class": kind if kind in EXCEPTIONS else "OTHER_ERROR_CLASS",
        "source_commit": SOURCE_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "model_skill_verified": False,
        "official_train_dev_ood_admitted": False,
        "blind_ood_proven": False,
        "private_media_or_raw_exception_saved": False,
    }


def write_private_failure(stage: str, err: BaseException) -> None:
    directory = Path("/kaggle/working")
    if (sys.platform != "linux" or not directory.is_dir()
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get(KERNEL_FLAG) != "1"):
        return
    path = directory / DIAG
    if not path.exists():
        path.write_text(json.dumps(
            safe_failure_receipt(stage, type(err).__name__),
            separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    global STAGE
    import torch
    hardware = verify_free(torch)
    root = Path("/tmp/hala-h324-private-beaconboard-original")
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
        from h319.beaconboard_dynamic_research import (
            SCENES, checked_source,
        )
        from h323.preloaded_dynamic_bridge import (
            run_preloaded_dynamic_source_research,
        )
        need(transformers.__version__ == "5.16.1"
             and len(SCENES) == 12
             and tuple(SCENES[i][1] for i in SCENE_INDICES) == ("en", "ar"))
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_source(html)
        os.environ.update(
            HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        STAGE = "MODEL_LOAD"
        torch.manual_seed(324)
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
        rows = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True, args=["--no-sandbox"])
            try:
                for index in SCENE_INDICES:
                    observed = run_preloaded_dynamic_source_research(
                        browser, html, processor, model, SCENES[index])
                    need(observed["status"] ==
                         "H323_ORIGINAL_EXPOSED_DYNAMIC_PRELOADED_BRIDGE_ONLY"
                         and observed["pretrained_model_identity_independently_attested"]
                             is False
                         and observed["source_public_exposed"] is True
                         and observed["official_train_dev_ood_admitted"] is False
                         and observed["blind_ood_proven"] is False
                         and observed["trained_model_gain_proven"] is False
                         and observed["can_release"] is False)
                    rows.append({
                        "language": SCENES[index][1],
                        "model_generations":
                            observed["real_model_generations_claimed_by_wrapper"],
                        "native_dispatched":
                            observed["native_actions_dispatched"],
                        "source_failure":
                            observed["source_categorical_failure"],
                        "task_success":
                            observed["source_original_task_success"],
                    })
            finally:
                browser.close()
        STAGE = "PRIVATE_AGGREGATE"
        receipt = verified_report(rows, hardware)
        (Path("/kaggle/working") / REPORT).write_text(
            json.dumps(receipt, separators=(",", ":"), sort_keys=True)
            + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as err:
        write_private_failure(STAGE, err)
        raise RuntimeError("H324 owner-private pretrained pilot stopped; "
                           "inspect ONLY the strict categorical report") from None

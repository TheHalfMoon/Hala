"""H265: private free Kaggle T4 real BASE Qwen3-VL screenshot->native action probe.

This is a FIRST-PARTY RESEARCH probe, not admission of TRAIN/DEV/OOD,
independent blind evaluation, fine-tuned Hala, or any release. No personal
screen, private gold, prior OOD, credential, or screenshot is uploaded.
The script must run in a PRIVATE, explicitly noncommercial Kaggle GPU kernel.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path, PurePath
import shutil
import subprocess
import sys

REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h264-screenshot-locked-native-inference"
SOURCE_SHA = "933a408c481694856b77f2126e34d927e4347d6b"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
TRANSFORMERS_VERSION = "5.16.1"
PLAYWRIGHT_VERSION = "1.58.0"
APP_SHA256 = "3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a"
RESULT_NAME = "hala-h265-private-firstparty-baseline.json"
RESULT_STATUS = "H265_REAL_PRETRAINED_FIRST_PARTY_RESEARCH_NOT_OOD"
MODEL_TASK = "Click the +1 button belonging to the first inventory item to add one unit. Return ONLY one JSON actions object with a single click or abstain action. Use screenshot pixels."
MAX_SECONDS = 120


def require_free_gpu_environment(root: Path, *, environ=None, torch=None) -> dict:
    """Guard REAL Kaggle T4 before any weight download (not a GPU emulator)."""
    env = os.environ if environ is None else environ
    if (not isinstance(root, PurePath) or not root.is_absolute()
            or str(root) != "/kaggle/working"
            or env.get("HALA_H265_NONCOMMERCIAL_FREE_GPU") != "1"
            or not str(env.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()
            or torch is None or not torch.cuda.is_available()):
        raise ValueError("H265 needs an explicitly opted-in real FREE Kaggle GPU kernel")
    devices = torch.cuda.device_count()
    if type(devices) is not int or devices < 1:
        raise ValueError("H265 needs a real CUDA device")
    names = tuple(torch.cuda.get_device_name(i) for i in range(devices))
    if not all("T4" in name for name in names):
        raise ValueError("H265 only free Kaggle NVIDIA T4, not paid/unknown hardware")
    return {"free_kaggle_gpu_verified": True, "visible_cuda_devices": devices,
            "gpu_family": "NVIDIA_T4"}


def validate_source_provenance(directory: Path, *, run: Callable = subprocess.run) -> Path:
    """Require exact upstream public H264 commit and first-party app bytes."""
    if not isinstance(directory, Path) or not directory.is_dir():
        raise ValueError("H265 missing repository source directory")
    completed = run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                    check=True, capture_output=True, text=True)
    if completed.stdout.strip() != SOURCE_SHA:
        raise ValueError("H265 downloaded unexpected Git head")
    changed = run(["git", "-C", str(directory), "status", "--porcelain"],
                  check=True, capture_output=True, text=True)
    if changed.stdout.strip():
        raise ValueError("H265 original source has modifications")
    file = directory / "h262" / "stockroom.html"
    if (not file.is_file() or file.is_symlink()
            or hashlib.sha256(file.read_bytes()).hexdigest() != APP_SHA256):
        raise ValueError("H265 first-party stockroom source changed")
    return file


def privacy_safe_measurement(*, generation_count: int, action_executed: bool,
                             action_kind: str | None, before: int, after: int,
                             env: dict, outcome: str) -> dict:
    """Aggregates only; no raw model text, GUI pixels or reference coordinates."""
    if (type(generation_count) is not int or generation_count not in (0, 1)
            or type(action_executed) is not bool
            or type(before) is not int or type(after) is not int
            or type(action_kind) not in (str, type(None))
            or action_kind not in (None, "click", "drag", "key_press",
                                  "type_text", "scroll", "abstain")
            or outcome not in ("ACTION_EXECUTED", "ABSTAIN", "INVALID_ACTION",
                               "POLICY_OR_BROWSER_ERROR")
            or env.get("free_kaggle_gpu_verified") is not True):
        raise ValueError("H265 result outside strictly bounded genuine run")
    if generation_count == 0 and outcome in ("ACTION_EXECUTED", "ABSTAIN"):
        raise ValueError("H265 forbids success claim without real generation")
    success = generation_count == 1 and action_executed and after == before + 1
    return {
        "status": RESULT_STATUS,
        "real_pretrained_model_generations": generation_count,
        "pinned_model_id": MODEL_ID,
        "pinned_model_revision": MODEL_REVISION,
        "source_git_sha": SOURCE_SHA,
        "source_first_party": True,
        "hardware": env,
        "native_action_kind": action_kind,
        "output_class": outcome,
        "native_action_dispatched": action_executed,
        "first_item_stock_incremented": after == before + 1,
        "first_party_research_task_success": success,
        "independent_third_party_ood": False,
        "old_consumed_h173_ood_read": False,
        "official_train_dev_ood_admitted": False,
        "new_hala_training_optimizer_steps": 0,
        "model_adapted_to_hala": False,
        "model_skill_qualified": False,
        "private_png_or_gold_saved": False,
    }


class CountedPreloadedModel:
    """Attest a completed generate call, not merely a Python callback call."""
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.completed_generations = 0

    def eval(self):
        self.model.eval()
        return self

    def generate(self, **kwargs):
        output = self.model.generate(**kwargs)
        self.completed_generations += 1
        return output


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_private_kaggle_probe() -> dict:
    if sys.platform != "linux":
        raise ValueError("H265 remote Linux only, NEVER download model to owner PC")
    root = Path("/kaggle/working")
    if not root.is_dir():
        raise ValueError("H265 expected Kaggle ephemeral workspace")
    # Kaggle's default torch must be a functioning CUDA build. Do not pip
    # install/replace Torch from a random package index.
    import torch
    platform = require_free_gpu_environment(root, torch=torch)
    if (root / RESULT_NAME).exists():
        raise ValueError("H265 create-only private report; refuse overwrite")
    checkout = root / "hala_h265_pinned_source"
    if checkout.exists():
        raise ValueError("H265 refuse reused or altered source directory")
    _run(["git", "clone", "--depth", "1", "--single-branch",
          "--branch", SOURCE_BRANCH, REPO, str(checkout)])
    stockroom = validate_source_provenance(checkout)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                      HF_HOME="/kaggle/working/hala_h265_hf_cache")
    # Exact known working Hala-era transformers version; original Kaggle
    # torch stays untouched. The model is never downloaded on owner devices.
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError("H265 exact transformers runtime not installed")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from playwright.sync_api import sync_playwright
    # Paths are placed AFTER source commit and SHA validation.
    sys.path.insert(0, str(checkout))
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h263.native_browser_executor import new_offline_context
    from h264.screenshot_locked_research import screenshot_locked_turn

    torch.manual_seed(265)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    core = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        trust_remote_code=False, attn_implementation="sdpa").to("cuda:0").eval()
    if getattr(core.config, "model_type", None) != "qwen3_vl":
        raise ValueError("H265 unexpected pretrained architecture")
    instrumented = CountedPreloadedModel(core)
    policy = PreloadedVisionPolicy(processor, instrumented, max_new_tokens=160)
    outcome = "POLICY_OR_BROWSER_ERROR"
    kind = None
    dispatched = False
    initial = final = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            context = new_offline_context(browser)
            try:
                context.route("**/*", lambda route: (
                    route.abort() if route.request.url.startswith(
                        ("http:", "https:", "ws:", "wss:"))
                    else route.continue_()))
                page = context.new_page()
                url = stockroom.as_uri()
                page.goto(url, wait_until="load")
                locator = page.locator(".item .count").first
                initial = int(locator.inner_text())
                # Only screenshot+task is accessible to the actual pretrained
                # model; DOM task-state oracle is evaluated separately here.
                try:
                    decision = screenshot_locked_turn(page, stockroom, url,
                                                      MODEL_TASK, policy)
                    kind = decision["kind"]
                    dispatched = decision["executed"]
                    outcome = ("ACTION_EXECUTED" if dispatched else "ABSTAIN")
                except (ValueError, RuntimeError):
                    # Do not print raw model response, PNG, prompt or private
                    # reference. Genuine failure is retained, NOT replaced
                    # with an invented/oracle-derived fallback click.
                    outcome = ("INVALID_ACTION" if
                               instrumented.completed_generations else
                               "POLICY_OR_BROWSER_ERROR")
                final = int(page.locator(".item .count").first.inner_text())
            finally:
                context.close()
        finally:
            browser.close()
    result = privacy_safe_measurement(
        generation_count=instrumented.completed_generations,
        action_executed=dispatched, action_kind=kind,
        before=initial, after=final, env=platform, outcome=outcome)
    (root / RESULT_NAME).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    # Never dump raw model outputs. All remote logs are aggregate-only.
    print(json.dumps(run_private_kaggle_probe(), sort_keys=True), flush=True)

"""H279: real pretrained Qwen normalized native action on a second original GUI family.

Private FREE Kaggle T4, author-owned Queueboard research only. No OOD,
admitted training data, adapter update, or generalized quality claim.
Never persist model text, screenshot, action points, DOM target or weights.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

SOURCE_REPO = "https://github.com/TheHalfMoon/Hala.git"
SOURCE_BRANCH = "ps/h278-queueboard-task-target-contract"
SOURCE_SHA = "c61409e0b0178227180f0d8e05c67f5a210d01f9"
SOURCE_BLOB = "3cf452e84a75b017905bc14b04a8571411929891"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
REPORT = "hala-h279-private-queueboard-native-aggregate.json"
KERNEL_FLAG = "HALA_H279_PRIVATE_FREE_NONCOMMERCIAL"
SCENES = (
    ("en_priority_q118", "en", True, "Q-118"),
    ("ar_all_q142", "ar", False, "Q-142"),
)
RELATIVE_SUFFIX = (
    " The image is 960x720 pixels. Choose the center of the correct "
    "visible control from the screenshot alone. Output Qwen3-VL native "
    "RELATIVE x,y coordinates, integers 0..999, where 0 is the left/top "
    "and 999 the right/bottom. Do not output pixel coordinates. "
    "Return exactly one JSON object with one click or abstain: "
    '{"actions":[{"kind":"click","point":[x,y]}]} or '
    '{"actions":[{"kind":"abstain"}]}. '
    "No Markdown, no other actions. Abstain if unsure."
)


class FormatRejected(ValueError):
    """Only strict model format failure; no raw text in exception."""


class RelativeDecodingProcessor:
    """Convert normalized Qwen output BEFORE H208's pixel-only H199 parser.

    H208 correctly validates *pixel* actions, but it cannot accept a raw
    0..999 normalized y coordinate above 719. Conversion must therefore
    happen in batch_decode, not after PreloadedVisionPolicy returns.
    """
    def __init__(self, base, converter):
        self.base = base
        self.converter = converter
        self.converted_calls = 0
        self.format_rejected = False

    def apply_chat_template(self, *args, **kwargs):
        return self.base.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        texts = self.base.batch_decode(*args, **kwargs)
        if type(texts) is not list or len(texts) != 1 or type(texts[0]) is not str:
            raise ValueError("H279 invalid decoded model continuation")
        try:
            native = self.converter(texts[0])
        except ValueError:
            self.format_rejected = True
            raise FormatRejected("H279 strict normalized model format rejected") from None
        self.converted_calls += 1
        return [native]


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H279 fail-closed: " + reason)


def source_task(scene: tuple) -> str:
    require(scene in SCENES, "unregistered firstparty scene")
    _, language, _, ticket = scene
    return (f"Complete fictional ticket {ticket}" if language == "en"
            else f"أكمل الطلب الخيالي {ticket}")


def validate_rows(rows: list, hardware: dict) -> dict:
    require(hardware.get("free_t4") is True and
            type(hardware.get("devices")) is int and
            hardware["devices"] in (1, 2), "unverified free GPU")
    require(type(rows) is list and len(rows) == len(SCENES),
            "two prospective real GUI rows required")
    for scene, row in zip(SCENES, rows):
        require(type(row) is dict and set(row) == {
            "scene", "real_generations", "grammar_generate_calls",
            "strict_normalized_json", "native_dispatched",
            "firstparty_task_success", "native_kind",
        }, "aggregate contains extra or missing fields")
        require(row["scene"] == scene[0] and
                type(row["real_generations"]) is int and
                row["real_generations"] == 1 and
                type(row["grammar_generate_calls"]) is int and
                row["grammar_generate_calls"] == 1 and
                all(type(row[k]) is bool for k in (
                    "strict_normalized_json", "native_dispatched",
                    "firstparty_task_success")) and
                row["native_kind"] in (None, "click", "abstain") and
                (row["strict_normalized_json"] or
                 (not row["native_dispatched"] and
                  not row["firstparty_task_success"] and
                  row["native_kind"] is None)) and
                (not row["native_dispatched"] or
                 row["strict_normalized_json"]) and
                (not row["firstparty_task_success"] or
                 (row["native_dispatched"] and row["native_kind"] == "click")),
                "unattested model/native outcome")
    return {
        "status": "H279_REAL_PRETRAINED_QUEUEBOARD_FIRSTPARTY_NOT_OOD",
        "source_commit": SOURCE_SHA, "source_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "hardware": hardware,
        "real_pretrained_image_generations": 2,
        "model_time_grammar_generate_calls": 2,
        "rows": rows,
        "firstparty_task_successes": sum(
            row["firstparty_task_success"] for row in rows),
        "optimizer_updates": 0,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "model_improvement_proven": False,
        "raw_model_text_points_screenshots_or_weights_published": False,
    }


def verify_free(torch) -> dict:
    root = Path("/kaggle/working")
    require(sys.platform == "linux" and root.is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get(KERNEL_FLAG) == "1"
            and torch.cuda.is_available()
            and torch.cuda.device_count() in (1, 2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "owner-private noncommercial free T4 only")
    require(shutil.disk_usage("/tmp").free >= 8 * 1024**3,
            "ephemeral model cache space insufficient")
    require(not (root / REPORT).exists(), "report must be create-only")
    return {"free_t4": True, "devices": torch.cuda.device_count()}


def clone_pinned_source(destination: Path) -> Path:
    require(not destination.exists(), "fresh ephemeral source checkout")
    subprocess.run(["git", "clone", "--depth", "1", "--single-branch",
                    "--branch", SOURCE_BRANCH, SOURCE_REPO, str(destination)],
                   check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(destination), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.strip()
    require(head == SOURCE_SHA and not dirty, "pinned source branch drift")
    html = destination / "h275" / "queueboard.html"
    require(html.is_file() and not html.is_symlink(),
            "original Queueboard source missing")
    data = html.read_bytes()
    blob = sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()
    require(blob == SOURCE_BLOB, "original source bytes drift")
    return html


def run() -> dict:
    import torch
    hardware = verify_free(torch)
    root = Path("/tmp/hala-h279-queueboard-pretrained")
    require(not root.exists(), "fresh ephemeral experiment required")
    root.mkdir(mode=0o700)
    try:
        html = clone_pinned_source(root / "source")
        sys.path.insert(0, str(html.parents[1]))
        import importlib.metadata
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from h268.free_t4_constrained_native import (
            NativeGrammarModel, verify_exact_lmfe_installed_source)
        from h273.relative_native_bridge import (
            normalize_strict_action, build_normalized_token_filter)
        from h276.queueboard_native_research import (
            checked_queueboard, run_firstparty_queueboard_probe)
        from h199.native_action_sft import source_task_prompt
        require(transformers.__version__ == "5.16.1",
                "pinned Transformers required")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_queueboard(html)
        os.environ.update(HF_HOME=str(root / "hf"),
                          HF_HUB_CACHE=str(root / "hf" / "hub"),
                          HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        torch.manual_seed(279)
        cache = str(root / "hf" / "hub")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(getattr(model.config, "model_type", None) == "qwen3_vl",
                "incorrect pretrained model")
        token_filter = build_normalized_token_filter(processor.tokenizer)
        rows = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                for scene in SCENES:
                    name, language, priority_only, ticket = scene
                    task = source_task(scene)
                    wrapped = NativeGrammarModel(model, token_filter)
                    decoder = RelativeDecodingProcessor(
                        processor, normalize_strict_action)
                    policy = PreloadedVisionPolicy(
                        decoder, wrapped, max_new_tokens=160)
                    def predict(png: bytes, prompt: str) -> str:
                        require(prompt == source_task_prompt(task),
                                "policy prompt identity drift")
                        # H208 sees already converted strict native PIXELS;
                        # raw relative model text never leaves decoder RAM.
                        try:
                            return policy(png, task + RELATIVE_SUFFIX)
                        except ValueError:
                            if decoder.format_rejected:
                                raise FormatRejected(
                                    "H279 strict normalized format rejected") from None
                            raise
                    valid = True
                    outcome = None
                    try:
                        outcome = run_firstparty_queueboard_probe(
                            browser, html, task, ticket, predict,
                            language=language, priority_only=priority_only)
                    except FormatRejected:
                        valid = False
                    require(wrapped.completed_generations == 1 and
                            wrapped.grammar_installed_calls == 1 and
                            decoder.converted_calls == (1 if valid else 0),
                            "real model/grammar and relative conversion not attested")
                    rows.append({
                        "scene": name,
                        "real_generations": wrapped.completed_generations,
                        "grammar_generate_calls": wrapped.grammar_installed_calls,
                        "strict_normalized_json": valid,
                        "native_dispatched": bool(outcome["native_action_dispatched"])
                            if outcome else False,
                        "firstparty_task_success": bool(outcome["firstparty_task_success"])
                            if outcome else False,
                        "native_kind": outcome["native_action_kind"]
                            if outcome else None,
                    })
            finally:
                browser.close()
        report = validate_rows(rows, hardware)
        (Path("/kaggle/working") / REPORT).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True), flush=True)

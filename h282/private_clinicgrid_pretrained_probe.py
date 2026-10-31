"""H282: prospectively pinned PRIVATE pretrained Qwen ClinicGrid research.

Original synthetic third GUI family (H281). This is NOT blind OOD,
training, checkpoint adaptation or release qualification.
Never persist raw model text, screenshots, image tensors, native
coordinates, user media, source gold or weights. A safe low-entropy
boolean-only aggregate is the sole permitted output.
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
SOURCE_BRANCH = "ps/h281-original-clinicgrid-browser-research"
SOURCE_SHA = "ab6d4c77ee89a6abfb4375edbba4f1a2108db445"
SOURCE_BLOB = "dacdfaf5c4b9fe9ed22624109c917f7e72bf4edb"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
REPORT = "hala-h282-private-clinicgrid-native-aggregate.json"
KERNEL_FLAG = "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL"
SCENES = (
    ("en_urgent_c214", "en", True, "C-214"),
    ("ar_all_c308", "ar", False, "C-308"),
)
RELATIVE_SUFFIX = (
    " The screenshot is 960x720 pixels. Choose the CENTER of the "
    "correct visible approval button only from the screenshot. "
    "Emit Qwen3-VL RELATIVE x,y integers from 0..999, "
    "with (0,0) at top left and (999,999) at bottom right. "
    "Never output absolute pixel coordinates. Output only one "
    'exact JSON object: {"actions":[{"kind":"click","point":[x,y]}]} '
    'or {"actions":[{"kind":"abstain"}]}. No Markdown, no extra '
    "actions or text. Abstain if unsure."
)


class FormatRejected(ValueError):
    """Safe categorical rejection, never echo private model text."""


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("H282 fail-closed: " + why)


def validate_rows(rows: object, hardware: object) -> dict:
    require(type(hardware) is dict and set(hardware) == {"free_t4", "devices"}
            and hardware["free_t4"] is True
            and type(hardware["devices"]) is int
            and hardware["devices"] in (1, 2), "not verified FREE T4")
    require(type(rows) is list and len(rows) == 2,
            "exactly two prospective model-scene results required")
    for i, row in enumerate(rows):
        require(type(row) is dict and set(row) == {
            "scene", "real_generations", "grammar_generate_calls",
            "screenshot_integrity_attested", "strict_normalized_json",
            "native_dispatched", "firstparty_task_success", "native_kind",
            "all_source_visits_checked",
        }, "row schema changed or includes private output")
        require(row["scene"] == SCENES[i][0]
                and type(row["real_generations"]) is int
                and row["real_generations"] == 1
                and type(row["grammar_generate_calls"]) is int
                and row["grammar_generate_calls"] == 1
                and type(row["screenshot_integrity_attested"]) is bool
                and row["screenshot_integrity_attested"] is row["strict_normalized_json"]
                and type(row["all_source_visits_checked"]) is int
                and row["all_source_visits_checked"] == 6
                and all(type(row[k]) is bool for k in (
                    "strict_normalized_json", "native_dispatched",
                    "firstparty_task_success"))
                and (row["native_kind"] is None
                     or type(row["native_kind"]) is str
                     and row["native_kind"] in ("click", "abstain")),
                "invalid genuine model/action evidence")
        require((row["strict_normalized_json"] or
                 (row["native_kind"] is None
                  and not row["native_dispatched"]
                  and not row["firstparty_task_success"]))
                and (not row["native_dispatched"]
                     or (row["strict_normalized_json"]
                         and row["native_kind"] == "click"))
                and (not row["firstparty_task_success"]
                     or (row["native_dispatched"]
                         and row["native_kind"] == "click")),
                "impossible native source-owned task success")
    return {
        "status": "H282_REAL_PRETRAINED_CLINICGRID_FIRSTPARTY_NOT_OOD",
        "source_commit": SOURCE_SHA,
        "source_html_git_blob": SOURCE_BLOB,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REV,
        "hardware": hardware,
        "real_pretrained_image_generations": 2,
        "model_time_grammar_generate_calls": 2,
        "rows": rows,
        "firstparty_task_successes": sum(
            int(row["firstparty_task_success"]) for row in rows),
        "optimizer_updates": 0,
        "independent_blind_ood": False,
        "official_train_dev_ood_admitted": False,
        "adapted_model_improvement_proven": False,
        "raw_model_text_points_screenshots_or_weights_published": False,
    }


def verify_free(torch) -> dict:
    working = Path("/kaggle/working")
    require(sys.platform == "linux" and working.is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and os.environ.get(KERNEL_FLAG) == "1"
            and torch.cuda.is_available()
            and torch.cuda.device_count() in (1, 2)
            and all("T4" in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())),
            "owner-private authorized noncommercial FREE T4 only")
    require(shutil.disk_usage("/tmp").free >= 8 * 1024**3,
            "insufficient ephemeral model cache")
    require(not (working / REPORT).exists(), "create-only report required")
    return {"free_t4": True, "devices": torch.cuda.device_count()}


def clone_source(destination: Path) -> Path:
    require(not destination.exists(), "fresh temporary source checkout")
    subprocess.run([
        "git", "clone", "--depth", "1", "--single-branch",
        "--branch", SOURCE_BRANCH, SOURCE_REPO, str(destination)
    ], capture_output=True, check=True, text=True)
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True, check=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain"],
        capture_output=True, check=True, text=True).stdout.strip()
    require(head == SOURCE_SHA and not dirty, "frozen clean source commit drift")
    html = destination / "h281" / "clinicgrid.html"
    require(html.is_file() and not html.is_symlink(),
            "pinned original GUI missing")
    raw = html.read_bytes()
    git_blob = b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw
    require(sha1(git_blob).hexdigest() == SOURCE_BLOB,
            "original ClinicGrid GUI byte drift")
    return html


def run() -> dict:
    import torch
    hardware = verify_free(torch)
    root = Path("/tmp/hala-h282-clinicgrid-pretrained")
    require(not root.exists(), "fresh ephemeral experiment")
    root.mkdir(mode=0o700)
    try:
        html = clone_source(root / "source")
        sys.path.insert(0, str(html.parents[1]))
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from playwright.sync_api import sync_playwright
        from h199.native_action_sft import source_task_prompt
        from h208.preloaded_vlm_policy import PreloadedVisionPolicy
        from h268.free_t4_constrained_native import (
            NativeGrammarModel, verify_exact_lmfe_installed_source)
        from h273.relative_native_bridge import (
            build_normalized_token_filter, normalize_strict_action,
            NORMALIZED_REGEX)
        from h279.private_relative_native_probe import (
            RelativePrevalidationProcessor)
        from h281.clinicgrid_native_research import (
            checked_clinicgrid, exact_task, run_original_clinicgrid_probe)

        require(transformers.__version__ == "5.16.1",
                "pinned Transformers 5.16.1 required")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        checked_clinicgrid(html)
        os.environ.update(
            HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        torch.manual_seed(282)
        cache = str(root / "hf" / "hub")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REV, cache_dir=cache,
            trust_remote_code=False, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        require(getattr(model.config, "model_type", None) == "qwen3_vl",
                "wrong Qwen pretrained model")
        token_filter = build_normalized_token_filter(processor.tokenizer)

        rows = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                for scene in SCENES:
                    task = exact_task(scene)
                    wrapped = NativeGrammarModel(model, token_filter)
                    decoder = RelativePrevalidationProcessor(
                        processor, normalize_strict_action, NORMALIZED_REGEX)
                    policy = PreloadedVisionPolicy(
                        decoder, wrapped, max_new_tokens=160)

                    def predict(png: bytes, prompt: str) -> str:
                        require(prompt == source_task_prompt(task),
                                "model screenshot instruction identity drift")
                        try:
                            return policy(png, task + RELATIVE_SUFFIX)
                        except ValueError:
                            if decoder.format_rejected:
                                raise FormatRejected(
                                    "H282 strict normalized format refused"
                                ) from None
                            raise

                    valid = True
                    result = None
                    try:
                        result = run_original_clinicgrid_probe(
                            browser, html, scene, predict)
                    except FormatRejected:
                        valid = False
                    require(wrapped.completed_generations == 1
                            and wrapped.grammar_installed_calls == 1
                            and decoder.converted_calls == int(valid),
                            "real model/grammar/strict decode count absent")
                    rows.append({
                        "scene": scene[0],
                        "real_generations": wrapped.completed_generations,
                        "grammar_generate_calls": wrapped.grammar_installed_calls,
                        "screenshot_integrity_attested": bool(result is not None),
                        "strict_normalized_json": valid,
                        "native_dispatched": (
                            result["native_action_dispatched"]
                            if result else False),
                        "firstparty_task_success": (
                            result["firstparty_task_success"]
                            if result else False),
                        "native_kind": (
                            result["native_action_kind"] if result else None),
                        "all_source_visits_checked": (
                            result["all_source_visits_checked"] if result else 6),
                    })
            finally:
                browser.close()
        report = validate_rows(rows, hardware)
        (Path("/kaggle/working") / REPORT).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    # Bounded only: no output of generated model strings / pixels / gold.
    print(json.dumps(run(), sort_keys=True), flush=True)

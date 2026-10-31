"""H287 owner-private FREE T4, real small LoRA + paired cross-source native actions.

TRAIN ONLY: H285/286 public original ShiftDesk candidates (24 validated,
8 preselected in balanced EN/AR for research optimizer updates).
EVAL ONLY: already exposed H275 Queueboard original; same two frozen
cases as H279 corrected V3. This is NOT blind OOD and NOT admitted
TRAIN/DEV/OOD, even if a positive paired source state delta appears.
All media, labels, weights and optimizer state stay in Kaggle private /tmp.
Only whitelisted small categorical research summary under /kaggle/working.
"""
from __future__ import annotations

from hashlib import sha1, sha256
from pathlib import Path
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys

REPO = "https://github.com/TheHalfMoon/Hala.git"
BRANCH = "ps/h286-relative-train-label-alignment"
PIN = "76a4ebb204d825b280fd4d332ad75591c5f13007"
EVAL_SOURCE_BRANCH = "ps/h278-queueboard-task-target-contract"
EVAL_SOURCE_SHA = "c61409e0b0178227180f0d8e05c67f5a210d01f9"
SHIFT_BLOB = "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac"
QUEUE_BLOB = "3cf452e84a75b017905bc14b04a8571411929891"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
REPORT = "hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json"
FLAG = "HALA_H292_PRIVATE_NONCOMMERCIAL_FREE_T4"
DIAGNOSTIC_REPORT = "hala-h292-private-routeboard-stage-diagnostic.json"
STAGE = "BOOTSTRAP"
DEV_SOURCE_BRANCH = "ps/h289-original-routeboard-dev-research"
DEV_SOURCE_SHA = "5a7608f533897ac9c10d41b3d97c4a0b245a000b"
DEV_SOURCE_BLOB = "6b7486384c935ff6ea856a0eefeb213a9504fdf2"
DEV_INDICES = (0, 1, 4, 6, 7, 10)
COMPOUND_REPORT = "hala-h292-private-routeboard-paired-aggregate.json"
COMPOUND_STATUS = "H292_REAL_FIRSTPARTY_SHIFT_TRAIN_QUEUE_ROUTE_DEV_RESEARCH_NOT_OOD"
STAGE_ENUM = frozenset({"BOOTSTRAP", "GPU_PREFLIGHT", "WORKSPACE", "SOURCE", "EVAL_SOURCE", "DEV_SOURCE", "DEPENDENCIES", "MODEL_LOADING", "TRAIN_CANDIDATES", "BASE_EVAL", "DEV_BASE", "LORA_SETUP", "OPTIMIZER", "ADAPTED_EVAL", "DEV_ADAPTED", "RESULT_VALIDATION", "RESULT_WRITE", "SUCCESS"})
STATUS = "H287_REAL_SHIFT_TRAIN_QUEUE_RESEARCH_NOT_ADMITTED_NOT_OOD"
STEP_INDICES = (0, 3, 6, 9, 12, 15, 18, 21)
N_STEPS = len(STEP_INDICES)
COORDINATE_WEIGHT = 8.0
QUEUE_SCENES = (
    ("en_priority_q118", "en", True, "Q-118"),
    ("ar_all_q142", "ar", False, "Q-142"),
)
# Freeze general, same train/evaluation model-facing spatial instruction.
# H286 emits H282-specific approval wording; replace ONLY this exact
# fixed input phrase, never transform a raw MODEL prediction.
FROM_PHRASE = "correct visible approval button"
TO_PHRASE = "correct visible action button"


def need(ok: bool, label: str) -> None:
    if not ok:
        raise ValueError("H287 fail-closed: " + label)


def verified_free_t4(torch) -> dict:
    need(sys.platform == "linux"
         and Path("/kaggle/working").is_dir()
         and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
         and os.environ.get(FLAG) == "1"
         and torch.cuda.is_available()
         and torch.cuda.device_count() in (1, 2)
         and all("T4" in torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count()))
         and shutil.disk_usage("/tmp").free >= 8 * 1024**3
         and not (Path("/kaggle/working") / REPORT).exists(),
         "authorized owner-private noncommercial FREE T4 only")
    return {"free_t4": True, "devices": torch.cuda.device_count()}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return sha1(b"blob " + str(len(data)).encode("ascii")
                + b"\0" + data).hexdigest()


def _verified_clone(branch: str, sha: str, target: Path) -> None:
    """Separate immutable Git source checkout with native cross-OS bytes."""
    need(not target.exists(), "fresh owner-private distinct source checkout")
    subprocess.run([
        "git", "-c", "core.autocrlf=false", "clone", "--depth", "1",
        "--single-branch", "--branch", branch, REPO, str(target),
    ], check=True, text=True, capture_output=True)
    actual = subprocess.run([
        "git", "-C", str(target), "rev-parse", "HEAD"
    ], check=True, text=True, capture_output=True).stdout.strip()
    dirty = subprocess.run([
        "git", "-C", str(target), "status", "--porcelain"
    ], check=True, text=True, capture_output=True).stdout.strip()
    need(actual == sha and not dirty,
         "pinned original source revision changed or checkout dirty")


def get_pinned_source(destination: Path) -> tuple[Path, Path]:
    """TRAIN H286 and EVAL H278 require DIFFERENT immutable Git revisions.

    H287 V1 and H290 V2 wrongly looked for both source families
    inside H286, which has ShiftDesk but NO H275 or H276 files.
    """
    global STAGE
    need(not destination.exists(), "new two-source private scratch required")
    destination.mkdir(mode=0o700)
    STAGE = "SOURCE"
    train_repo = destination / "train_h286"
    _verified_clone(BRANCH, PIN, train_repo)
    train_html = train_repo / "h285" / "shiftdesk.html"
    need(train_html.is_file() and not train_html.is_symlink()
         and _git_blob(train_html) == SHIFT_BLOB,
         "TRAIN ShiftDesk source blob not pinned original")

    STAGE = "EVAL_SOURCE"
    eval_repo = destination / "eval_h278"
    _verified_clone(EVAL_SOURCE_BRANCH, EVAL_SOURCE_SHA, eval_repo)
    eval_html = eval_repo / "h275" / "queueboard.html"
    eval_module = eval_repo / "h276" / "queueboard_native_research.py"
    need(eval_html.is_file() and not eval_html.is_symlink()
         and _git_blob(eval_html) == QUEUE_BLOB
         and eval_module.is_file() and not eval_module.is_symlink(),
         "EVAL original Queueboard HTML and independent source-owned oracle missing")
    return train_html, eval_html


def model_prompt(text: str) -> str:
    """Only pretrained prompt text; never repair model output."""
    from h282.private_clinicgrid_pretrained_probe import RELATIVE_SUFFIX
    need(type(text) is str and text.endswith(RELATIVE_SUFFIX)
         and text.count(FROM_PHRASE) == 1,
         "H286 trained label prompt must match pinned relative syntax")
    corrected = text.replace(FROM_PHRASE, TO_PHRASE)
    need(corrected.endswith(RELATIVE_SUFFIX.replace(FROM_PHRASE, TO_PHRASE))
         and corrected.count(TO_PHRASE) == 1,
         "source-disjoint train/eval model prompt construction changed")
    return corrected


def queue_task(scene: tuple) -> str:
    need(type(scene) is tuple and scene in QUEUE_SCENES,
         "pre-registered firstparty research only")
    return ("Complete fictional ticket " + scene[-1]
            if scene[1] == "en" else
            "أكمل الطلب الخيالي " + scene[-1])


def validate_training_evidence(info: dict, count: int, en: int, ar: int) -> None:
    from h285.train_only_browser_candidates import ROLE
    need(type(info) is dict
         and info.get("role") == ROLE
         and info.get("original_source_blob") == SHIFT_BLOB
         and info.get("relative_targets_roundtrip_verified") == 24
         and info.get("actual_browser_supervisions") == 24
         and info.get("english_candidates") == 12
         and info.get("arabic_candidates") == 12
         and info.get("real_model_generations") == 0
         and info.get("official_train_admitted") is False
         and info.get("source_disjoint_blind_ood") is False
         and count == 24 and en == 12 and ar == 12,
         "original 24 source-only TRAIN candidates not qualified")


def prepare_private_train(
    browser, source: Path, root: Path,
) -> tuple[Path, list[dict], frozenset[str]]:
    """Materialize ONLY in owner-private ephemeral Kaggle tmp for trainer."""
    from h285.train_only_browser_candidates import ROLE, SCENES
    from h286.relative_candidate_training import produce_relative_train_candidates
    media = root / "private_train_png"
    need(not media.exists(), "TRAIN scratch is create-only")
    media.mkdir(mode=0o700)
    records = []
    seen = set()
    def accept(png: bytes, prompt: str, completion: str) -> None:
        index = len(records)
        need(index < 24, "unexpected extra TRAIN example")
        name, lang, *_ = SCENES[index]
        need(type(png) is bytes and png.startswith(b"\x89PNG\r\n\x1a\n")
             and type(prompt) is str and type(completion) is str
             and len(png) < 4 * 1024 * 1024,
             "untrusted TRAIN candidate")
        aligned = model_prompt(prompt)
        from h273.relative_native_bridge import normalize_strict_action
        native = normalize_strict_action(completion)
        from h199.native_action_sft import parse_actions
        need(parse_actions(native, training_target=True)["actions"][0]["kind"] == "click",
             "invalid aligned relative TRAIN label")
        digest = sha256(png).hexdigest()
        need(digest not in seen, "duplicate TRAIN PNG")
        seen.add(digest)
        # This private ephemeral output MUST be deleted at experiment end.
        path = media / ("private_train_%02d.png" % index)
        need(not path.exists(), "duplicate private training file")
        path.write_bytes(png)
        records.append({
            "scene": name, "language": lang,
            "id": "h287-private-train-%02d" % index,
            "image_path": path.name, "image_sha256": digest,
            "image_format": "PNG",
            "prompt": aligned, "completion": completion,
        })
    try:
        info = produce_relative_train_candidates(
            browser, source, accept, role=ROLE)
        en = sum(r["language"] == "en" for r in records)
        ar = sum(r["language"] == "ar" for r in records)
        validate_training_evidence(info, len(records), en, ar)
        need(tuple(r["language"] for r in (records[i] for i in STEP_INDICES))
             == ("en", "ar", "en", "ar", "en", "ar", "en", "ar"),
             "balanced eight-step order drift")
        return media, records, frozenset(seen)
    except BaseException:
        shutil.rmtree(media, ignore_errors=True)
        raise


class StrictFormatRefused(ValueError):
    """Private raw model text never included in error strings."""


def evaluate_queueboard_pair(
    browser, source: Path, processor, model, grammar, forbidden_hashes,
) -> tuple[list[dict], tuple[str, ...]]:
    from h199.native_action_sft import source_task_prompt
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h268.free_t4_constrained_native import NativeGrammarModel
    from h273.relative_native_bridge import (
        normalize_strict_action, NORMALIZED_REGEX)
    from h279.private_relative_native_probe import RelativePrevalidationProcessor
    from h276.queueboard_native_research import (
        checked_queueboard, run_firstparty_queueboard_probe)
    from h282.private_clinicgrid_pretrained_probe import RELATIVE_SUFFIX

    checked_queueboard(source)
    rows, frames = [], []
    for scene in QUEUE_SCENES:
        name, lang, priority, target = scene
        task = queue_task(scene)
        wrapped = NativeGrammarModel(model, grammar)
        decoder = RelativePrevalidationProcessor(
            processor, normalize_strict_action, NORMALIZED_REGEX)
        policy = PreloadedVisionPolicy(decoder, wrapped, max_new_tokens=160)
        captures = []
        def predict(png: bytes, prompt: str) -> str:
            need(prompt == source_task_prompt(task)
                 and len(captures) == 0
                 and png.startswith(b"\x89PNG\r\n\x1a\n"),
                 "exactly one fresh screenshot-only task call")
            digest = sha256(png).hexdigest()
            need(digest not in forbidden_hashes,
                 "TRAIN PNG leaked into source-disjoint research EVAL")
            captures.append(digest)
            try:
                return policy(png, task + RELATIVE_SUFFIX.replace(
                    FROM_PHRASE, TO_PHRASE))
            except ValueError:
                if decoder.format_rejected:
                    raise StrictFormatRefused("H287 strict relative format refused") from None
                raise
        valid = True
        outcome = None
        try:
            outcome = run_firstparty_queueboard_probe(
                browser, source, task, target, predict,
                language=lang, priority_only=priority)
        except StrictFormatRefused:
            valid = False
        need(len(captures) == 1
             and wrapped.completed_generations == 1
             and wrapped.grammar_installed_calls == 1
             and decoder.converted_calls == int(valid),
             "real image-conditioned model-time grammar not attested")
        frames.extend(captures)
        rows.append({
            "scene": name,
            "model_generation": 1,
            "grammar_call": 1,
            "strict": valid,
            "kind": outcome["native_action_kind"] if outcome else None,
            "native_click": bool(outcome["native_action_dispatched"])
                if outcome else False,
            "task_success": bool(outcome["firstparty_task_success"])
                if outcome else False,
        })
    need(len(set(frames)) == 2, "paired EVAL scenes not visibly distinct")
    return rows, tuple(frames)


def verified_report(
    steps: list[dict], base: list[dict], adapted: list[dict],
    before: tuple[str, ...], after: tuple[str, ...],
    platform: dict, train_hashes: frozenset[str],
) -> dict:
    need(platform.get("free_t4") is True
         and type(platform.get("devices")) is int
         and platform["devices"] in (1, 2),
         "real private T4 evidence missing")
    need(type(train_hashes) is frozenset and len(train_hashes) == 24,
         "real 24 distinct TRAIN candidate screenshots absent")
    need(type(steps) is list and len(steps) == N_STEPS
         and all(type(row) is dict
                 and set(row) == {"changed", "coordinates_supervised",
                                 "loss_finite", "gradient_nonzero"}
                 and type(row["changed"]) is int and row["changed"] > 0
                 and row["coordinates_supervised"] is True
                 and row["loss_finite"] is True
                 and row["gradient_nonzero"] is True
                 for row in steps), "genuine finite changed LoRA steps absent")
    need(type(before) is tuple and type(after) is tuple
         and len(before) == len(after) == 2
         and before == after
         and len(set(before)) == 2
         and set(before).isdisjoint(train_hashes),
         "paired screenshots changed or TRAIN/EVAL image leakage")
    for mode in (base, adapted):
        need(type(mode) is list and len(mode) == 2
             and tuple(x.get("scene") for x in mode)
             == tuple(s[0] for s in QUEUE_SCENES),
             "paired research source case drift")
        for row in mode:
            need(type(row) is dict and set(row) == {
                "scene", "model_generation", "grammar_call",
                "strict", "kind", "native_click", "task_success",
            }, "row contains raw/private field")
            need(type(row["model_generation"]) is int
                 and row["model_generation"] == 1
                 and type(row["grammar_call"]) is int
                 and row["grammar_call"] == 1
                 and all(type(row[k]) is bool
                         for k in ("strict", "native_click", "task_success"))
                 and (row["kind"] is None or
                      type(row["kind"]) is str
                      and row["kind"] in ("click", "abstain"))
                 and (row["strict"] or (not row["native_click"]
                      and not row["task_success"] and row["kind"] is None))
                 and (not row["native_click"] or
                      (row["strict"] and row["kind"] == "click"))
                 and (not row["task_success"] or row["native_click"]),
                 "fabricated trained/pretrained research outcome")
    b = sum(x["task_success"] for x in base)
    a = sum(x["task_success"] for x in adapted)
    return {
        "status": STATUS, "source_commit": PIN,
        "train_original_blob": SHIFT_BLOB,
        "eval_original_blob": QUEUE_BLOB,
        "model_id": MODEL, "model_revision": MODEL_REV,
        "hardware": platform,
        "original_train_candidates_verified": 24,
        "research_train_optimizer_steps": N_STEPS,
        "changed_lora_tensors_each_step": True,
        "coordinate_supervision_each_step": True,
        "paired_original_eval_scenes": 2,
        "paired_pretrained": base,
        "paired_adapted": adapted,
        "paired_pretrained_successes": b,
        "paired_adapted_successes": a,
        "paired_delta": a - b,
        "same_exact_eval_screens_before_after": True,
        "train_eval_exact_png_disjoint": True,
        "research_source_exposed_not_blind_ood": True,
        "official_train_dev_ood_admitted": False,
        "generalizable_hala_improvement_proven": False,
        "adapter_weights_published": False,
        "raw_predictions_images_points_or_gold_published": False,
    }


def evaluate_routeboard_dev(
    browser, source: Path, processor, model, grammar, forbidden_train_hashes,
) -> tuple[list[dict], tuple[str, ...]]:
    """Six pre-registered *EXPOSED* original DEV scenes, real native Qwen.

    DOM state is for original source-owned post-action scoring ONLY.
    The model's single fresh input is screenshot + frozen task text.
    """
    from h199.native_action_sft import source_task_prompt
    from h208.preloaded_vlm_policy import PreloadedVisionPolicy
    from h268.free_t4_constrained_native import NativeGrammarModel
    from h273.relative_native_bridge import (
        normalize_strict_action, NORMALIZED_REGEX)
    from h279.private_relative_native_probe import RelativePrevalidationProcessor
    from h282.private_clinicgrid_pretrained_probe import RELATIVE_SUFFIX
    from h289.routeboard_dev_research import (
        SCENES as DEV_SCENES, ROLE as DEV_ROLE,
        checked_source, exact_task, run_original_dev_probe)
    checked_source(source)
    need(len(DEV_INDICES) == 6 and len(set(DEV_INDICES)) == 6,
         "original RouteBoard DEV protocol selection drift")
    chosen = tuple(DEV_SCENES[i] for i in DEV_INDICES)
    need(sum(row[1] == "en" for row in chosen) == 3
         and sum(row[1] == "ar" for row in chosen) == 3,
         "balanced DEV source language preregistration missing")
    rows, frames = [], []
    for scene in chosen:
        task = exact_task(scene)
        wrapped = NativeGrammarModel(model, grammar)
        decoder = RelativePrevalidationProcessor(
            processor, normalize_strict_action, NORMALIZED_REGEX)
        policy = PreloadedVisionPolicy(
            decoder, wrapped, max_new_tokens=160)
        captures = []
        def predict(png: bytes, prompt: str) -> str:
            need(prompt == source_task_prompt(task)
                 and len(captures) == 0
                 and type(png) is bytes
                 and png.startswith(b"\x89PNG\r\n\x1a\n"),
                 "one fresh DEV screenshot-only policy input")
            digest = sha256(png).hexdigest()
            need(digest not in forbidden_train_hashes,
                 "original TRAIN screenshot leaked into exposed DEV")
            captures.append(digest)
            try:
                return policy(png, task + RELATIVE_SUFFIX.replace(
                    FROM_PHRASE, TO_PHRASE))
            except ValueError:
                if decoder.format_rejected:
                    raise StrictFormatRefused(
                        "H292 prospective strict Qwen relative format refused"
                    ) from None
                raise
        valid = True
        result = None
        try:
            result = run_original_dev_probe(
                browser, source, scene, predict, role=DEV_ROLE)
        except StrictFormatRefused:
            valid = False
        need(len(captures) == 1
             and wrapped.completed_generations == 1
             and wrapped.grammar_installed_calls == 1
             and decoder.converted_calls == int(valid),
             "real DEV image-conditioned model-time grammar not attested")
        frames.extend(captures)
        rows.append({
            "scene": scene[0],
            "language": scene[1],
            "model_generation": 1, "grammar_call": 1,
            "strict": valid,
            "kind": result["native_kind"] if result else None,
            "native_click": bool(result["native_action_dispatched"])
                if result else False,
            "task_success": bool(result["firstparty_task_success"])
                if result else False,
        })
    need(len(rows) == 6 and len(set(frames)) == 6,
         "duplicate original DEV research image or missing task")
    return rows, tuple(frames)


def run() -> dict:
    global STAGE
    STAGE = "GPU_PREFLIGHT"
    import torch
    platform = verified_free_t4(torch)
    STAGE = "WORKSPACE"
    root = Path("/tmp/hala-h292-original-routeboard-paired")
    need(not root.exists(), "fresh ephemeral experiment required")
    root.mkdir(mode=0o700)
    try:
        STAGE = "SOURCE"
        train_html, eval_html = get_pinned_source(root / "source")
        STAGE = "DEV_SOURCE"
        dev_repo = root / "source" / "dev_h289"
        _verified_clone(DEV_SOURCE_BRANCH, DEV_SOURCE_SHA, dev_repo)
        dev_html = dev_repo / "h289" / "routeboard.html"
        need(dev_html.is_file() and not dev_html.is_symlink()
             and _git_blob(dev_html) == DEV_SOURCE_BLOB,
             "independent original RouteBoard DEV source blob missing")
        sys.path.insert(0, str(dev_html.parents[1]))
        sys.path.insert(0, str(eval_html.parents[1]))
        sys.path.insert(0, str(train_html.parents[1]))
        STAGE = "DEPENDENCIES"
        import transformers
        import peft
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        from playwright.sync_api import sync_playwright
        from h268.free_t4_constrained_native import (
            verify_exact_lmfe_installed_source)
        from h273.relative_native_bridge import build_normalized_token_filter
        from training.scripts.train_adapter import (
            prepare_inputs, LORA_TARGET_PATTERN)
        from h61.supervision import weighted_lm_loss

        need(transformers.__version__ == "5.16.1"
             and peft.__version__ == "0.20.0",
             "exact pinned vision model/trainer dependency drift")
        verify_exact_lmfe_installed_source(
            importlib.metadata.distribution("lm-format-enforcer"))
        os.environ.update(
            HF_HOME=str(root / "hf"),
            HF_HUB_CACHE=str(root / "hf" / "hub"),
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        torch.manual_seed(287)
        STAGE = "MODEL_LOADING"
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=MODEL_REV,
            cache_dir=str(root / "hf" / "hub"), trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, revision=MODEL_REV,
            cache_dir=str(root / "hf" / "hub"), trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
        need(getattr(model.config, "model_type", None) == "qwen3_vl",
             "wrong real pretrained Qwen model")
        grammar = build_normalized_token_filter(processor.tokenizer)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                STAGE = "TRAIN_CANDIDATES"
                media, records, train_hashes = prepare_private_train(
                    browser, train_html, root)
                STAGE = "BASE_EVAL"
                base, base_frames = evaluate_queueboard_pair(
                    browser, eval_html, processor, model, grammar, train_hashes)
                STAGE = "LORA_SETUP"
                STAGE = "DEV_BASE"
                base_dev, base_dev_frames = evaluate_routeboard_dev(
                    browser, dev_html, processor, model, grammar, train_hashes)
                need(set(base_dev_frames).isdisjoint(base_frames),
                     "Queueboard research images overlap with RouteBoard DEV")
                STAGE = "LORA_SETUP"
                lora = LoraConfig(
                    r=4, lora_alpha=8,
                    target_modules=LORA_TARGET_PATTERN,
                    lora_dropout=0, task_type="CAUSAL_LM", bias="none")
                model = get_peft_model(model, lora)
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                model.config.use_cache = False
                params = [p for p in model.parameters() if p.requires_grad]
                need(bool(params) and
                     sum(p.numel() for p in params) <= 2_000_000,
                     "unexpected pretrained weights were unfrozen")
                # Match existing Hala FP16 adapter train scalar assumptions.
                for p in params:
                    p.data = p.data.float()
                optimizer = torch.optim.AdamW(params, lr=5e-5)
                scaler = torch.amp.GradScaler("cuda", init_scale=1.0, enabled=True)
                steps = []
                STAGE = "OPTIMIZER"
                model.train()
                for index in STEP_INDICES:
                    row = records[index]
                    batch, labels, weights = prepare_inputs(
                        row, processor, media, 8192,
                        supervision="coordinate_weighted",
                        coordinate_weight=COORDINATE_WEIGHT,
                        return_weights=True)
                    coordinate_tokens = int((weights > 1.0).sum().item())
                    need(coordinate_tokens > 0 and
                         int((labels != -100).sum().item()) >= coordinate_tokens,
                         "missing supervised normalized coordinate tokens")
                    batch = batch.to("cuda:0")
                    labels = labels.to("cuda:0")
                    weights = weights.to("cuda:0")
                    old_params = [p.detach().cpu().clone() for p in params]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = weighted_lm_loss(
                            model(**batch).logits, labels, weights)
                    need(bool(torch.isfinite(loss).item()),
                         "nonfinite true FP16 vision-language loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient = float(
                        torch.nn.utils.clip_grad_norm_(params, 1.0).item())
                    need(math.isfinite(gradient) and gradient > 0,
                         "nonfinite/zero adapter gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    changed = sum(
                        not torch.equal(old, p.detach().cpu())
                        for old, p in zip(old_params, params))
                    need(changed > 0, "no LoRA weights actually updated")
                    steps.append({
                        "changed": changed, "coordinates_supervised": True,
                        "loss_finite": True, "gradient_nonzero": True,
                    })
                    del batch, labels, weights, old_params, loss
                    torch.cuda.synchronize(0)
                model.eval()
                model.config.use_cache = True
                STAGE = "ADAPTED_EVAL"
                adapted, adapted_frames = evaluate_queueboard_pair(
                    browser, eval_html, processor, model, grammar, train_hashes)
                STAGE = "RESULT_VALIDATION"
                queueboard_report = verified_report(
                    steps, base, adapted, base_frames, adapted_frames,
                    platform, train_hashes)
                STAGE = "DEV_ADAPTED"
                adapted_dev, adapted_dev_frames = evaluate_routeboard_dev(
                    browser, dev_html, processor, model, grammar, train_hashes)
                need(base_dev_frames == adapted_dev_frames,
                     "paired DEV screenshots changed between BASE and ADAPTED")
                dev_base_success = sum(r["task_success"] for r in base_dev)
                dev_adapted_success = sum(r["task_success"] for r in adapted_dev)
                report = {
                    "status": COMPOUND_STATUS,
                    "original_queueboard_research": queueboard_report,
                    "original_dev_source_commit": DEV_SOURCE_SHA,
                    "original_dev_html_blob": DEV_SOURCE_BLOB,
                    "original_dev_exposed_not_blind_ood": True,
                    "dev_scenes_preregistered": len(DEV_INDICES),
                    "dev_model_frames_distinct_and_train_queue_disjoint": True,
                    "paired_dev_screenshots_identical": True,
                    "dev_base": base_dev, "dev_adapted": adapted_dev,
                    "dev_base_successes": dev_base_success,
                    "dev_adapted_successes": dev_adapted_success,
                    "dev_paired_delta": dev_adapted_success-dev_base_success,
                    "official_train_dev_ood_admitted": False,
                    "generalized_hala_skill_proven": False,
                    "raw_predictions_images_points_gold_or_weights_saved": False,
                }
            finally:
                browser.close()
        STAGE = "RESULT_WRITE"
        out = Path("/kaggle/working") / COMPOUND_REPORT
        need(not out.exists(), "private report create-only")
        out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                       encoding="utf-8")
        STAGE = "SUCCESS"
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


def failure_receipt(stage: str, exception_class: str) -> dict:
    """Bounded stage-only diagnostics; no raw exception messages or data."""
    allowed_classes = frozenset({
        "RuntimeError", "ValueError", "TypeError", "ImportError",
        "ModuleNotFoundError", "FileNotFoundError", "OSError",
        "AssertionError", "KeyError", "IndexError", "AttributeError",
        "MemoryError", "OutOfMemoryError", "CalledProcessError",
    })
    return {
        "status": "H292_PRIVATE_FAILURE_DIAGNOSTIC_NOT_A_MODEL_SCORE",
        "source_protocol": "H287_ORIGINAL_SHIFT_TRAIN_QUEUE_RESEARCH",
        "previous_v1_terminal_error": True,
        "failure_stage": stage if stage in STAGE_ENUM else "BOOTSTRAP",
        "exception_class": (exception_class if exception_class in allowed_classes
                            else "OTHER_ERROR_CLASS"),
        "model_id": MODEL, "model_revision": MODEL_REV,
        "optimizer_steps_verified": 0,
        "paired_gui_task_score_verified": False,
        "official_training_admitted": False,
        "independent_blind_ood_verified": False,
        "model_improvement_proven": False,
        "raw_exception_logs_images_points_or_weights_saved": False,
    }


def write_private_failure_receipt(stage: str, exc: Exception) -> None:
    """Never write a diagnostic outside authenticated owner-private Kaggle."""
    if (sys.platform != "linux" or not Path("/kaggle/working").is_dir()
            or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get(FLAG) != "1"):
        return
    target = Path("/kaggle/working") / DIAGNOSTIC_REPORT
    if target.exists():
        return
    document = failure_receipt(stage, type(exc).__name__)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if len(payload.encode("utf-8")) > 1200:
        return
    target.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True), flush=True)
    except Exception as exc:
        write_private_failure_receipt(STAGE, exc)
        raise RuntimeError("H292 private trial aborted; retrieve only "
                           "the exact stage-only diagnostic JSON") from None

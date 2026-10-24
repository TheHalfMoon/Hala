"""H95: paired frozen real H92 vs H94 adapters, new official DragOn EVAL cases.

Exactly 4 NEW original public-EVAL scenes per domain (indices 4..7), source
pinned. Official strict H61 scorer is unchanged; fenced JSON normalization
is a clearly separate diagnostic. Does not train or claim release quality.
"""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from h61.official_eval import (
    DOMAINS, REPO, REV, apply_frame, canonical, fetch_domain, load_scenes, score)
from h85.model_cache import HF_HOME, pinned_cache_kwargs
from h93.eval import (
    _infer, admit_h92_adapter, H92_ADAPTER_SHA, H92_CHECKPOINT_SHA)
from h94.research_train import research_configuration, STEPS
from h73.kaggle_h73_cpu import TRAIN_SHA
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

H94_KERNEL = "hala-h94-private-h92-warmstart-qwen-64-steps"
H94_ADAPTER_SHA = "d73cc9a302782bf7eca08f434a2050a9c3f212368c25afe65014f71319eb673c"
H94_CHECKPOINT_SHA = "c13fb4ce619318075006af3ac24f5fd0ce43757cb5746201c9d595328d3594ac"
START_INDEX = 4
PER_DOMAIN = 4
FETCH_COUNT = START_INDEX + PER_DOMAIN
MAX_GENERATED_TOKENS = 128
SCHEMA = "hala.h95.private-new-eval-paired-h92-h94-v1"


def require_private_eval(*, env=None, system=None, work=None, output=None):
    env = os.environ if env is None else env
    system = platform.system() if system is None else system
    if (system != "Linux" or not env.get("KAGGLE_KERNEL_RUN_TYPE")
            or env.get("HALA_NONCOMMERCIAL_RESEARCH") != "1"
            or work is None or output is None
            or not str(work).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")
            or Path(output).exists()):
        raise ValueError("private noncommercial read-only Kaggle evaluation required")


def admit_h94_adapter(root: Path, repo: Path) -> Path:
    """Exact original private H94 complete optimizer+adapter SHA readback."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("private Kaggle kernel source root required")
    candidates = [
        path.parent for path in root.rglob("latest.json")
        if not path.is_symlink()
        and path.parent.name == "hala-h94-private-checkpoint"
        and H94_KERNEL in path.as_posix()
    ]
    if len(candidates) != 1:
        raise ValueError("exactly one authentic private H94 kernel output required")
    version, state = load_checkpoint(candidates[0], None)
    adapter = version / "adapter"
    weights = adapter / "adapter_model.safetensors"
    config = research_configuration(repo)
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()
    signed_digest = hashlib.sha256(
        (config_digest + H92_ADAPTER_SHA).encode()).hexdigest()
    if (not weights.is_file() or weights.is_symlink()
            or not (adapter / "adapter_config.json").is_file()
            or sha256_file(weights) != H94_ADAPTER_SHA
            or sha256_file(version / "checkpoint.json") != H94_CHECKPOINT_SHA
            or state.get("step") != STEPS or state.get("max_steps") != STEPS
            or state.get("dataset_sha256") != TRAIN_SHA
            or state.get("config_sha256") != signed_digest
            or state.get("changed_parameter_tensors", 0) <= 0):
        raise ValueError("H94 adapter identity / pinned source not qualified")
    return adapter


def normalize_fenced_json(raw):
    """Format-only diagnostic: do not alter original strict benchmark scoring."""
    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("```json\n") and value.endswith("\n```"):
            return value[8:-4]
    return raw


def run(*, repo: Path, work: Path, output: Path,
        attached_inputs: Path = Path("/kaggle/input")) -> dict:
    require_private_eval(work=work, output=output)
    adapter_h92 = admit_h92_adapter(attached_inputs, repo)
    adapter_h94 = admit_h94_adapter(attached_inputs, repo)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    chosen = {}
    for domain in DOMAINS:
        folder = work / domain
        fetch_domain(domain, FETCH_COUNT, folder)
        scenes = apply_frame(load_scenes(folder / "scenes.json"), folder)
        if len(scenes) != FETCH_COUNT:
            raise ValueError("bounded fixed new DragOn sample incomplete")
        chosen[domain] = scenes[START_INDEX:START_INDEX + PER_DOMAIN]
        if len(chosen[domain]) != PER_DOMAIN:
            raise ValueError("new paired sample count mismatch")
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if (not torch.cuda.is_available() or transformers.__version__ != "5.16.1"):
        raise ValueError("real provider GPU and pinned transformers are required")
    os.environ.update(HF_HOME=HF_HOME, HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, **pinned_cache_kwargs(MODEL_ID, MODEL_REVISION))
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, **pinned_cache_kwargs(MODEL_ID, MODEL_REVISION),
        torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
    model = PeftModel.from_pretrained(
        base, adapter_h92, adapter_name="h92", is_trainable=False,
        trust_remote_code=False).eval()
    summary = {}
    raw_proofs = {}
    started = time.monotonic()
    for label in ("h92", "h94"):
        if label == "h94":
            model.load_adapter(
                str(adapter_h94), adapter_name="h94",
                is_trainable=False, trust_remote_code=False)
        model.set_adapter(label)
        model.eval()
        summary[label] = {}
        raw_proofs[label] = {}
        for domain in DOMAINS:
            scenes = chosen[domain]
            outputs, proofs = _infer(model, processor, scenes, torch)
            strict = score(scenes, outputs)
            normalized = score(
                scenes, {key: normalize_fenced_json(value)
                         for key, value in outputs.items()})
            fields = (
                "scenes", "action_valid", "start_hit", "end_hit",
                "task_success", "task_success_rate",
                "degenerate_start_equals_end")
            summary[label][domain] = {
                "strict": {k: strict[k] for k in fields},
                "format_only_diagnostic": {k: normalized[k] for k in fields}
            }
            raw_proofs[label][domain] = proofs
            print("H95_REAL_PAIRED_NEW_SCENES", label, domain, len(scenes), flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "private-raw-neural-attempts.json"
    raw_path.write_bytes(canonical(raw_proofs))
    count = {label: sum(summary[label][domain]["strict"]["scenes"] for domain in DOMAINS)
             for label in ("h92", "h94")}
    strict_successes = {
        label: sum(summary[label][domain]["strict"]["task_success"] for domain in DOMAINS)
        for label in ("h92", "h94")}
    normalized_successes = {
        label: sum(summary[label][domain]["format_only_diagnostic"]["task_success"]
                   for domain in DOMAINS)
        for label in ("h92", "h94")}
    if count != {"h92": 16, "h94": 16}:
        raise ValueError("paired new official sample count mismatch")
    report = {
        "schema": SCHEMA,
        "status": "REAL_PAIRED_NEW_OFFICIAL_PUBLIC_EVAL_DIAGNOSTIC",
        "source_repo": REPO, "source_commit": REV, "source_split": "eval",
        "source_selection": "fixed original image ordering indices 4..7, per domain",
        "sample_counts": count,
        "strict_successes": strict_successes,
        "strict_delta_h94_minus_h92": strict_successes["h94"] - strict_successes["h92"],
        "format_only_successes_diagnostic": normalized_successes,
        "per_domain_scores": summary,
        "h92_adapter_sha256": H92_ADAPTER_SHA,
        "h92_checkpoint_sha256": H92_CHECKPOINT_SHA,
        "h94_adapter_sha256": H94_ADAPTER_SHA,
        "h94_checkpoint_sha256": H94_CHECKPOINT_SHA,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "raw_evidence_sha256": sha256_file(raw_path),
        "training_executed_in_eval": False,
        "official_strict_score_modified": False,
        "statistically_reliable_full_benchmark": False,
        "bilingual_model_quality_qualified": False,
        "commercial_use": False,
        "public_model_released": False,
        "elapsed_seconds": round(time.monotonic() - started, 1)
    }
    (output / "h95-private-eval-summary.json").write_bytes(canonical(report))
    return report

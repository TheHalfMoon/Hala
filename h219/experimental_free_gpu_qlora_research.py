"""H219: experimental ONE-OR-TWO-step PUBLIC Qwen3-VL QLoRA probe.

Research-only: H212 entirely self-authored synthetic TRAIN *candidates*.
Never call H199 verify_real_training_release, never admit official data,
touch original/quarantined owner screenshots, or evaluate consumed OOD.
Real GPU execution has NOT yet been performed/qualified. No publication,
checkpoint, uploads, external tracking, private tokens or saved weights.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
RESEARCH_INDICES = (0, 1)
ACKNOWLEDGEMENT = "SELF_AUTHORED_RESEARCH_ONLY_NOT_HALА_OFFICIAL"
MIN_VRAM_BYTES = 14 * 1024**3
MAX_VRAM_BYTES = 25 * 1024**3


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("H219 free GPU research-only block: " + reason)


def source_policy(indices: tuple[int, ...] = RESEARCH_INDICES) -> dict:
    """No neural deps, no owner media, no Pillow, no model or training."""
    from h199.native_action_sft import ADMITTED_TRAIN_MANIFEST_SHAS
    from h212.procedural_native_gui_candidates import COUNT, enumerate_specs
    require(not ADMITTED_TRAIN_MANIFEST_SHAS,
            "official original GUI TRAIN allowlist unexpectedly opened")
    require(type(indices) is tuple and indices == RESEARCH_INDICES,
            "one fixed new bilingual training CANDIDATE pair only")
    roster = enumerate_specs()
    require(COUNT == 96 and len(roster) == 96,
            "H212 original source identity changed")
    for index in indices:
        _idx, role, family, variant = roster[index]
        require(_idx == index and role == "RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED"
                and family == "toolbar" and variant == index,
                "not new unadmitted procedural toolbar research cases")
    return {
        "kind": "EXPERIMENT_ONLY_NOT_OFFICIAL_TRAINING",
        "synthetic_examples": len(indices),
        "language_assignments": ("en", "ar"),
        "official_train_approved": False,
        "independent_dev_sealed": False,
        "true_ood_consumed": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }


def offline_owner_machine_block() -> dict:
    """Safe no-model check on a constrained owner Windows desktop."""
    result = source_policy()
    require(result["optimizer_steps"] == 0
            and result["official_train_approved"] is False,
            "no owner CPU/GPU research step allowed")
    return result


def check_runtime_request(*, acknowledged: str, free_session: bool,
                          gpu_count: int, vram_bytes: int,
                          output_dir: Path, cwd: Path,
                          token_present: bool) -> None:
    """Check before downloading public weights or importing PEFT/trainer."""
    require(acknowledged == ACKNOWLEDGEMENT,
            "explicit nonpublishing experiment acknowledgment required")
    require(free_session is True and token_present is False,
            "public free GPU session without private HF token required")
    require(gpu_count == 1 and type(vram_bytes) is int
            and MIN_VRAM_BYTES <= vram_bytes <= MAX_VRAM_BYTES,
            "one free 14-25GiB CUDA GPU, never a laptop CPU")
    require(isinstance(output_dir, Path)
            and output_dir == Path("/kaggle/working/Hala-H219-Research-Only")
            and isinstance(cwd, Path) and cwd.is_absolute()
            and Path("/kaggle/working").is_dir(),
            "ephemeral nonpublished Kaggle working directory required")


def _verify_real_processor_identity(processor) -> None:
    from transformers import Qwen3VLProcessor
    require(type(processor) is Qwen3VLProcessor,
            "real Qwen3-VL model processor class required")


def _fetch_public_processor_assets(directory: Path) -> Path:
    """Download only 8 public assets from exact revision, never user data."""
    import shutil
    from huggingface_hub import snapshot_download
    from h218.pinned_qwen_processor_bytes import (
        PINNED_PUBLIC_FILES, SNAPSHOT_REVISION, verify_pinned_snapshot,
    )
    source = Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=list(PINNED_PUBLIC_FILES), token=False,
    ))
    proc_root = directory / "processor" / SNAPSHOT_REVISION
    require(not proc_root.exists(), "must use an empty ephemeral processor dir")
    proc_root.mkdir(parents=True)
    for name in PINNED_PUBLIC_FILES:
        p = source / name
        require(p.is_file(), "public pinned model processor asset unavailable")
        shutil.copyfile(p, proc_root / name)
    verify_pinned_snapshot(proc_root)
    return proc_root


def real_experimental_qlora_two_steps() -> list[dict]:
    """At most two real optimizer steps; never run during CI or on Windows."""
    from h213.frozen_candidate_reproduction_gate import verify_checked_in_source
    source_policy()
    source_receipt = verify_checked_in_source()
    require(source_receipt["candidate_cases_checked"] == 96
            and source_receipt["rgba_sha_matches"] == 96
            and source_receipt["official_train_admitted"] is False,
            "public self-authored PNG/RGBA source identity fails")
    # Delayed imports: no torch/network/model in source-only tests.
    import torch
    gpu_count = torch.cuda.device_count()
    available_vram = (
        torch.cuda.get_device_properties(0).total_memory if gpu_count == 1 else 0
    )
    root = Path("/kaggle/working/Hala-H219-Research-Only")
    check_runtime_request(
        acknowledged=os.environ.get("HALA_H219_EXPLICIT_ACK", ""),
        free_session=os.environ.get("HALA_H219_FREE_GPU_SESSION") == "1",
        gpu_count=gpu_count, vram_bytes=available_vram,
        output_dir=root, cwd=Path.cwd(),
        token_present=any(os.environ.get(k) for k in
                          ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")),
    )
    import transformers
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen3VLForConditionalGeneration)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from h212.procedural_native_gui_candidates import research_multimodal_messages
    from h199.native_action_sft import assistant_completion_mask
    # Avoid raw prompts, image bytes, labels, tokens or any checkpoint logs.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(219)
    proc_root = _fetch_public_processor_assets(root)
    processor = AutoProcessor.from_pretrained(
        str(proc_root), local_files_only=True, trust_remote_code=False,
    )
    _verify_real_processor_identity(processor)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=False,
        quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, trust_remote_code=False,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    require(0 < len(params) < 1200 and sum(p.numel() for p in params) > 0,
            "Qwen LoRA trainable parameter set unexpectedly absent")
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    receipts = []
    from h216.real_qwen_processor_alignment import audit_sample
    for i in RESEARCH_INDICES:
        # On the same actual pinned processor verify zero-leak mask/vision
        # parity before each independent research-only gradient step.
        observed = audit_sample(processor, i)
        require(observed["assistant_native_click_roundtrip"]
                and observed["vision_values_identical"],
                "actual visual assistant target not safe to supervise")
        question, answer = research_multimodal_messages(i)
        kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
        prompt = processor.apply_chat_template(
            question, add_generation_prompt=True, **kwargs)
        full = processor.apply_chat_template(
            answer, add_generation_prompt=False, **kwargs)
        prefix = prompt["input_ids"][0].tolist()
        full_ids = full["input_ids"][0].tolist()
        mask = assistant_completion_mask(prefix, full_ids)
        require(mask[:len(prefix)] == [-100]*len(prefix)
                and all(t >= 0 for t in mask[len(prefix):]),
                "cannot train on image/user tokens")
        labels = torch.tensor([mask], dtype=torch.long, device="cuda:0")
        full = full.to("cuda:0")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**full, labels=labels).loss
        require(bool(torch.isfinite(loss).item()),
                "nonfinite supervised research loss before backward")
        before = params[0].detach().float().cpu().clone()
        loss.backward()
        require(any(p.grad is not None and bool(torch.isfinite(p.grad).all().item())
                    and bool((p.grad != 0).any().item()) for p in params),
                "no finite nonzero gradient on LoRA parameters")
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        after = params[0].detach().float().cpu()
        # In some LoRA configs the first tensor may not update on step 1;
        # instead explicitly require any LoRA tensor to have valid grads.
        require(bool(torch.isfinite(after).all().item()),
                "LoRA parameter became nonfinite after optimizer step")
        receipts.append({
            "step": i + 1, "language": "en" if i == 0 else "ar",
            "supervised_tokens": len(mask)-len(prefix),
            "finite_loss": bool(torch.isfinite(loss).item()),
            "finite_lora_gradients": True,
            "peak_cuda_gib": round(
                torch.cuda.max_memory_reserved() / (1024**3), 2
            ),
            "official_training": False, "eval_on_ood": False,
            "checkpoint_saved": False,
        })
        # Never log target JSON, owner media, user text or raw model outputs.
        print("H219_RESEARCH_ONLY_REAL_LORA_STEP",
              "step="+str(i+1), "language="+receipts[-1]["language"],
              "finite_loss=true", "official_training=false",
              "checkpoint_saved=false", flush=True)
        del full, prompt, labels, loss
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H219 experimental 2-step QLoRA only on public self-authored GUIs"
    )
    parser.add_argument("--source-only-preflight", action="store_true")
    parser.add_argument("--run-free-gpu-research-two-steps", action="store_true")
    args = parser.parse_args()
    require(args.source_only_preflight != args.run_free_gpu_research_two_steps,
            "choose exactly one source-only or explicit GPU research mode")
    if args.source_only_preflight:
        receipt = offline_owner_machine_block()
        print("H219_SOURCE_RESEARCH_CANDIDATES_CHECKED",
              "cases="+str(receipt["synthetic_examples"]),
              "optimizer_steps=0 official_train=0 owner_media=0")
        return
    try:
        result = real_experimental_qlora_two_steps()
    except Exception:
        raise SystemExit("H219_REAL_GPU_RESEARCH_STEP_BLOCKED_NO_RELEASE") from None
    require(len(result) == 2 and all(r["finite_loss"] for r in result),
            "research-only pair incomplete")
    print("H219_REAL_RESEARCH_TWO_STEPS_COMPLETED",
          "optimizer_steps=2 official_train=0 sealed_dev=0 ood=0",
          "checkpoint_saved=0 no_publication=1")


if __name__ == "__main__":
    main()

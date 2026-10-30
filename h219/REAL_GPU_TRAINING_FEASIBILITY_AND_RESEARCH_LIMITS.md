# H219 — first real training-step experiment: source prepared, execution not claimed

**Draft-only stacked on H218 PR #239 exact `516c21a5b37f13e110541fc3e0d3a1f2c4444fd9`.**
No merge, model/adapter publication, cloud export of owner/private data, formal TRAIN admission,
consumed H173 OOD or reserved Gantt access.

## The actual blocker, not a hypothetical

On owner Windows (2026-10-30), the observed CPU is 12 logical cores,
16 GB RAM (15.7 GiB reported) with **only 4.2 GiB free physical RAM**,
**7.46 GiB D free** and **3.74 GiB C free**; graphics are Intel UHD only.
No NVIDIA CUDA GPU is available. Consequently a genuine Qwen3-VL 2B QLoRA
optimizer step on that owner computer is not a safe or valid target.
Do not fetch full 2B weights onto this constrained owner machine.

H216 confirmed **real**, offline Qwen3VLProcessor parity on all 96
self-authored PNG screenshots; H218 subsequently measured and
pinned **all 8 public processor files (11,499,725 bytes)**. H219 is the
first purpose-built **actual gradient research protocol**, not
another tokenizer/source-only parity gate. But H219 can make no
claim of a completed gradient step until an eligible free NVIDIA
GPU session actually performs one.

## Public evidence and implementation basis

Official Hugging Face TRL Qwen3-VL QLoRA sample:
`https://github.com/huggingface/trl/blob/main/examples/sft_qwen3_vl/sft_qwen3_vl.ipynb`.
Hugging Face PEFT quantization guide:
`https://huggingface.co/docs/peft/developer_guides/quantization`.
The open-weight public upstream `Qwen/Qwen3-VL-2B-Instruct` is Apache-2.0.
The prior examples show 4-bit NF4 and LoRA; none establishes H219 fit
or accuracy on a particular Kaggle free GPU.

## Deliberately narrow research protocol

Source: `h219/experimental_free_gpu_qlora_research.py`, hardcoded
`MODEL_ID=Qwen/Qwen3-VL-2B-Instruct`, exact upstream model/processor
revision `89644892e4d85e24eaac8bacfd4f463576704203`.

- **One explicit 14–25GiB CUDA GPU and an acknowledged zero-cost
  Kaggle notebook runtime**. Python source-only preflight works
  without NVIDIA, torch, model download or processor.
- Requires exact research-only acknowledgement and proof (operator-
  supplied runtime flag) of free GPU access; this cannot prove
  whether any external account incurs a charge. Never run without
  confirming the free session.
- Selects **only H212 indices 0 and 1**, originally self-authored
  toolbar screenshots, one English and one Arabic. Both have role
  `RESEARCH_TRAIN_CANDIDATE_NOT_ADMITTED`. All historical owner
  images, 20 quarantined old originals, 16 candidate DEV, 16 extra
  research scenes, prior consumed OOD, true gold and reserved Gantt
  are excluded. These two are **not official TRAIN**.
- Checks H215 original frozen SHA metadata / all 96 RGBA hashes on
  Linux before model/weight download. Assembles a **fresh exact
  8-file public processor folder** from the upstream pinned revision,
  verifies all eight file lengths/digests against H218 original
  Windows receipt, and requires the real `Qwen3VLProcessor` class.
- Loads **only the public base model** at pinned revision into
  4-bit NF4, 16-bit compute, LoRA r=4, q_proj/v_proj; uses
  `prepare_model_for_kbit_training`, freezes other weights,
  AdamW and bounded gradient clipping. This is an
  *experimental engineering feasibility probe*, not
  a production checkpoint.
- Before each actual step, reprocesses the full PNG/PIL image
  through real H216 validation, checks prompt/full token-prefix
  identity and vision tensors and strictly masks all original
  prompt/vision tokens with `-100`; only 20-ish native JSON
  assistant tokens are supervised. If visual data is missing or
  loss is non-finite, **abort without reporting success**.
- At most **two** actual optimizer steps and no validation.
  No output adapter, no checkpoint, no `push_to_hub`, no
  TensorBoard, no W&B, no Trackio, no loss or raw labels exported.
  Prints only step/language/finite-loss and max GPU VRAM.
- A successful 2-step experiment **is not a trained Hala
  model**, does not prove generalization, and cannot qualify
  official TRAIN/DEV/OOD or authorize model release.
- Failure (e.g. session GPU unavailable, OOM, model
  interface mismatch) is an honest blocker, not a successful step.

## Tested and untested

`python -m h219.experimental_free_gpu_qlora_research --source-only-preflight`
checks research-only identity in pure Python with no neural imports,
downloads or owner media.
`tests/test_h219_experimental_qlora_preflight.py` rejects altered
source indices, official data promotion, unconfirmed free session,
multi-GPU/low-VRAM/CPU owner runs, auth token use, unexpected
output roots and wrong explicit acknowledgement. Hosted CI only
runs those pure source tests on Linux/Windows Python 3.11/3.12.
It cannot certify the actual QLoRA optimizer path.

**As authored, the real GPU experiment has NOT run, and zero real Hala
optimizer steps have been observed.** This draft is an auditable
research-only entry point, not evidence of actual model improvement.
Do not upload private repository source into any public Kaggle notebook;
use only a private session and original self-authored generator code,
or keep execution blocked. Do not activate automatic notebook publishing.

## Remaining Hala completion gates

Source/fork/old-RGBA historical completeness, independent
sealed DEV, clean novel OOD, pretrained weight
qualification, actual measured real optimizer steps,
strict native-action model eval on fresh private-safe
scenes, native real browser performance and Windows
release qualification are all still missing.
`h199.ADMITTED_TRAIN_MANIFEST_SHAS` stays EMPTY.

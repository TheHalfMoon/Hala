# H94 — 64-step H92-seeded Qwen LoRA research experiment

**Scope:** Kaggle account owner's private and noncommercial research, no paid GPU, no model publication, no PR merge.

## Why

H87 successfully produced **one** genuine Qwen3-VL-2B LoRA optimizer update and SHA-reopened private checkpoint. H92 warm-started H87's adapter for **eight** additional optimizer updates with fresh AdamW state. Owner-private host recomputed all 8 checkpoint versions and confirmed 26/26 byte-integrity tests, including pinned original H71 TRAIN SHA.

H93 actually ran **Qwen base vs H92** on an identical 16-scene official public DragOn EVAL sample (4 × 4 domains): unchanged strict scorer **0/16 for both**. All 32 generated responses were fenced Markdown JSON, explaining original parse failure, but format-only normalization (diagnostic, NOT official score) yielded **15/16 base valid actions and 14/16 H92 valid actions**, still **0/16 successful geometry** for either. Seven of 16 per arm produced degenerate start=end drags. Original official scene geometry and raw responses were SHA-reconciled on owner Windows. Geometry miss median ~434 px in both arms. This demonstrates a genuine grounding problem, not a proven improvement from H92.

## Bounded training experiment

H94 keeps Qwen model pin, original immutable H71 source-only TRAIN, matching original H61 curriculum, FP16, rank-8 LoRA and gradient accumulation=1. A new 64-step optimizer starts **from SHA-proven private H92 adapter**, independently certified by H93's unchanged adapter source admission. It **does not resume H92 AdamW state** and makes no claim of 73 contiguous optimizer steps. The only config change from source-verified H92 is `max_steps=64`. Every optimizer step must show positive gradient and changed parameters, finite positive loss; original immutable checkpoint publisher persists adapter+optimizer and private receipt. GPU selection is Kaggle T4 with separate VRAM per card.

The immutable code SHA fetched by this notebook is `c6aa0e6bef4e87638eaaf22af15af77d875ddd29`. No private weights, dataset, raw examples or credentials enter public GitHub.

## Submission and qualification

The private Kaggle notebook metadata is `h94/kernel-metadata.json`, attaching only owner-private H71 train dataset and H92 source notebook:

```powershell
kaggle kernels push -p C:\Users\Shehr\work\Hala\kaggle-h94-private-run --accelerator NvidiaTeslaT4 --timeout 2400
kaggle kernels status abdulazizshehri/hala-h94-private-h92-warmstart-qwen-64-steps
```

Notebook `COMPLETE` alone is insufficient. Retrieve `hala-h94-training-receipt.json`, `hala-h94-attempt.json` and private `hala-h94-private-checkpoint/`; validate the immutable optimizer and safetensors SHA **on the owner host** before claiming 64 real optimizer steps. Next compare a fixed independent DEV and public EVAL sample against **both Qwen base and H92** with unchanged strict scorer and a separately labeled fence-format diagnostic; do not tune prompt or select examples using official EVAL. No bilingual score or release readiness is implied by this training run. Keep all PRs Draft/NO-GO.

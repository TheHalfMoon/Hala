# H426 — H425 trained-model synthetic baseline, private GPU handoff

Status on **2026-11-07**: **BLOCKED BY OWNER KAGGLE FREE GPU WEEKLY QUOTA**, not a completed evaluation. The owner Kaggle CLI rejected the private GPU kernel push with `Maximum weekly GPU quota of 30.00 hours reached.` No new H425 inference was run, no H426 model was trained, and no synthetic-model accuracy should be quoted. The old H425 GPU evaluations are unaffected.

## Purpose

Read-only pilot of the **authentic private H425 trained LoRA** against newly invented H426 v4 DEV screenshots, with two different English/Arabic prompts for each of six images (two per domain; 12 inference actions). This is author-visible development data, **not independently sealed OOD**. No clinical safety or release claim is possible. This is not a H426 training experiment.

## Exact source-only packaging and privacy

- `h426/kaggle_h425_readonly_synthetic.py` verifies the existing H425 source-qualified training receipt and the known original H425 adapter SHA **before** staging a read-only LoRA; loads the official Qwen3-VL-2B immutable revision; and checks all eight public processor files byte-for-byte with H218/H387.
- Reuses the frozen H425 original parsing and scoring code, and the existing H426 synthetic DEV paired scorer. No optimizer, training step, model-save, original training JPEG, raw prediction, or private adapter bytes are published.
- `h426/build_private_readonly_kaggle.py` packages only 12 source-code files into a deterministic, hash-verified private Kaggle script. The generated script and kernel metadata are **outside** the repository on the owner's D: drive. Metadata forces `is_private=true`, free Tesla T4, and exactly the existing private H425-trained-parent dataset. Public Qwen base weights can download while model evaluation runs offline.
- The runtime validates the exact aggregate schema, H425 model hashes, per-domain counts and nonrelease flags with `h426/verify_private_h425_baseline.py` before any create-only output. Source-only regression coverage is in `tests/test_h426_verify_private_h425_baseline.py`.
- Runtime result is create-only and aggregate-only: per-domain English/Arabic and both-correct counts, number of images, authenticated weight/receipt hashes, no individual responses. An incomplete or failed GPU kernel must **not** be represented as a successful evaluation.

Owner source-only builder:

```powershell
Set-Location D:\Hala-H425-Worktree
& D:\Hala-H426-Tokenizer-Private\venv\Scripts\python.exe h426\build_private_readonly_kaggle.py --output-directory D:\Hala-H426-Private-H425-Synthetic-ReadOnly-20261107-v2
```

The destination uses **create-only** output to prevent accidental replacement. To retry later, first inspect the existing owner-private folder and use a **new**, explicitly reviewed private destination if a regenerated source is needed. Do not silently remove existing Kaggle results or overwrite private receipts. A real provider run still requires available free GPU quota. Retry from the designated private folder with `kaggle kernels push -p <owner-private-folder>` only when the quota is available and after verifying the source/metadata exact hashes.

## Qualification gates

1. Source-only package tests: `python -m unittest tests.test_h426_private_readonly_kaggle`.
2. Owner provider receipt must show a **COMPLETED** private GPU kernel and a validated aggregate of six images and 12 EN/AR actions. The Kaggle quota rejection is an explicit *no-go*, not evidence of an accuracy regression.
3. Validate reported source and adapter SHA, exact prompt/image identities, and frozen original H425 score. Only then compare any future H426 candidate on the same images.
4. No H426 training on original rendered screenshot assets before independent ML rights review [#384](https://github.com/TheHalfMoon/Hala/issues/384), and no production release without external blind bilingual OOD custody [#385](https://github.com/TheHalfMoon/Hala/issues/385).

This step is **source/test readiness only** as of 2026-11-07.

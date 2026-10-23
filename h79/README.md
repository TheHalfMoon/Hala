# H79 — Source-first admission before package installation

**Parent:** H78 Draft PR #62, source code SHA `623c698c8cfcfc6ef2890bb0061cc513ff510a3a`.

The previous remote H62 launcher installed and import-tested pinned Python packages **before** checking if the private H71 CPU-prepared TRAIN ZIP was present, byte-perfect and source-qualified. On a short-lived free Colab T4, absent or corrupt staged data would waste scarce compute time on packages without permitting a single optimizer step.

H79 moves the **already implemented** standard-library-only H67 `restore_stage(..., rederive=True)` before package checking and `ensure_packages()`. The original H67 validator still proves operator ZIP SHA256, H71 source Git SHA, complete TRAIN-only inventory, source and image digest matches and independent H61 curriculum rederivation. Its result must explicitly carry `RESTORED_TRAIN_STAGE` and match both expected source and ZIP hashes. Otherwise the launcher fails closed **before touching pip or model weights**.

No source identities, original H71 archive, train/eval split, or software dependencies were changed. For legacy non-staged diagnostics the original package-first flow remains intact. H65-H78 checkpoint/training safeguards and Kaggle GPU-training restriction are unchanged.

The H79 tests emulate a free session without contacting Colab or Kaggle. They verify successful restored-source admission happens before package pinning, missing/wrong/corrupt ZIP or rederivation errors cause **zero install calls**, incorrect restore metadata is rejected, and the non-staged path remains supported.

**Not model evidence:** zero real GPU steps, zero real model checkpoints, no provider publication/readback and no GUI performance measurement from H79. After availability of a truly free eligible T4, the workflow still needs pinned Qwen loading, first optimizer update, H65 host checkpoint proof, optionally H77 independent private authenticated readback, and held-out bilingual GUI assessment.

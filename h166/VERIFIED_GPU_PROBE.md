# H166 — independently qualified synthetic-only real free-T4 vision-merger LoRA probe

**2026-10-28. Real Kaggle NVIDIA Tesla T4 CUDA step completed and independently audited.**
**Feasibility only: NOT new GUI-task accuracy, not a trained Hala release.**

## Exact running evidence

- Private FREE Kaggle notebook `abdulazizshehri/hala-h166-private-visual-merger-t4-probe`, version 1, returned KernelWorkerStatus.COMPLETE.
- Job source pinned to Git SHA `ea6b6f5cf471e2e48896a8f6cd4c74035607cc84`.
- Base model: `patrickballou/gui-assist-grounder`, revision `31632edaa20749c7f532868a42635f7240033106` (2B external research only; Hala 35B unchanged).
- Pinned environment: torch 2.11.0, transformers 5.17.0, peft 0.21.2, Pillow 11.3.0.
- Private kernel had **zero attached datasets**. The *only* input image was a deterministic 960×720 synthetic white canvas with a red square, created by the probe code. It used no original app images, private gold targets or consumed OOD/DEV.
- Exactly **one real optimizer step** with rank-4 alpha-8 LoRA on these verified target paths (and NO language layers):
  - `model.visual.merger.linear_fc1`
  - `model.visual.merger.linear_fc2`
- Trainable LoRA parameters: **57,344**; **2 nonzero finite gradient tensors**, **2 updated parameter tensors**, **4 optimizer states**, finite synthetic loss.
- Actual Tesla T4 CUDA peak **allocated 10.950 GiB** (required <=13.0 GiB); peak **reserved 11.443 GiB**. The allocation margin to the predeclared safety cap was **2.050 GiB**. This is NOT guaranteed free VRAM for future multi-step or combined-adapter runs.
- GPU receipt status `H166_ONE_SYNTHETIC_VISUAL_MERGER_STEP_PASSED`; job status `H166_SYNTHETIC_FREE_T4_MEMORY_STEP_PASS`.
- Owner-independent `python -m h166.audit_probe --output <PRIVATE_GPU_OUTPUT>` reproduced `H166_INDEPENDENT_FREE_T4_VISUAL_MERGER_SYNTHETIC_PROBE_PASS`, cross-checking two distinct output receipts, pinned code/base/version, target allowlist, finite CUDA figures, optimizer proof and all safety flags.

## Boundary and decision

**GO for the next separate fresh-source TRAIN-only design/probe — not GO for model quality or deployment.**
A synthetic-only visual-merger gradient step demonstrates feasibility of a narrow adapter in this one GPU environment. It says **nothing** about whether spatial grounding, Arabic instructions, browser-native click outcomes, cross-app drag or typing, or overall reliability will improve. It does **not** prove the combined H164 language LoRA + visual adapter will fit under the same cap.

Before real training: qualify independently new licensed original application TRAIN scenes (train-only bilingual varied layouts/actions) and a disjoint new DEV, pin SHA and zero-leak provenance; perform compatibility and memory preflight of the frozen language baseline plus visual adapter. Preregister entirely new OOD app families, real native state reward and per-app/language pass criteria before model inference. Any past H140/H162/H163/H164 OOD source is fully consumed; do not reopen/tune/replay it.

Raw Kaggle receipts kept owner-private at `D:\Hala-H166-VisionProbe-20261028`; only aggregate non-sensitive results in GitHub. No trained weights saved/published, no H140 blind calls, no paid GPU, no flagship 35B modification, no main merge and no release.

# H167b — verified two-adapter frozen-language + visual LoRA free-T4 compatibility

**2026-10-28, owner-private Kaggle version 1, independently verified.**
**Research compatibility and memory result ONLY, not native GUI reward or real training.**

## Previously blocked H167

Original H167 was unqualified with `TypeError: unhashable type: 'list'` in
`peft_model.py:1622` before any optimizer step. PEFT `PeftModel.set_adapter`
accepts a single name; use `model.base_model.set_adapter(["language","vision"])`
for the LoRA tuner's multi-adapter selection. New fail-closed tests prevent
calling the incompatible wrapper. Failure receipt retained, not overwritten.

## Frozen input and real CUDA experiment

- Owner-private FREE Kaggle T4 kernel:
  `abdulazizshehri/hala-h167b-private-peft-dual-adapter-t4`,
  status `KernelWorkerStatus.COMPLETE`.
- Executed code source SHA:
  `ae69beef93521d060a2d8a9f5620785b21f55c1f`.
- Research base model:
  `patrickballou/gui-assist-grounder`, pinned revision
  `31632edaa20749c7f532868a42635f7240033106`.
- Only Kaggle input: private previously trained frozen H164 language-LoRA
  files. Exact ZIP SHA `4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da`;
  safetensors SHA `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
  No original app media, training-gold, DEV or OOD attached.
- Actual active adapter list: `["language","vision"]`.
  Frozen language LoRA **5,455,872 params**, digest identical before/after.
  Rank-4 merger visual LoRA **57,344 trainable params**, only
  `visual.merger.linear_fc1` and `visual.merger.linear_fc2`.
- Exactly **two synthetic optimizer steps** on an original-free white/red-square
  image: English (2 nonzero gradient tensors, 2 changed) and Arabic
  (4 nonzero gradient tensors, 4 changed). Both finite losses.
- Actual Tesla T4 CUDA peak: **7.025 GiB allocated**,
  **9.506 GiB reserved**. Predeclared allocated cap **13.0 GiB** passed.
  Differences from H166 peak **10.950 GiB** are not a controlled
  optimization comparison; H167b enables checkpointing.
- Private job result: `H167_SYNTHETIC_COMBINED_FREE_T4_PASS`.
  Independent `python -m h167b.audit --output <OWNER_PRIVATE_OUTPUT>`
  returned `H167B_INDEPENDENT_DUAL_ADAPTER_SYNTHETIC_T4_PASS`,
  checking separately generated job/model receipts, frozen source
  SHA, all input weights/config hashes, parameter counts and
  update evidence, language-specific steps, CUDA cap and
  no owner-gold/OOD/no-save flags.
- Owner Windows H16-focused regression **128 passed**, using D-drive
  pytest base temp to avoid known Windows permission restrictions.
  H167b adds four regression tests covering the precise PEFT failure.

## Quality/release boundary

**GO for source qualification of fresh bilingual TRAIN/DEV data, NOT GO
for task quality or release.** Combined-adapter feasibility does not
establish better Arabic localization, new-app clicks, browser state rewards,
typing, or drags. H165's new-app native test remains 1/64→10/64 and
Arabic only 1/32; those app screenshots are consumed and cannot be
reused for evaluation, hyperparameter tuning, training or selection.
No Hala 35B changes, no paid GPU, no new OOD calls, no published
weights and no merge to `main`.

CI maintenance: GitHub Ubuntu unittest discovers modules containing
`import pytest`, so pin the free pytest dev/test dependency and add an
explicit H16 pytest pass. Do not treat the earlier CI errors from absent
pytest as product behavior; require the corrected GitHub check to pass.

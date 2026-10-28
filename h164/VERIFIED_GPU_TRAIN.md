# H164 — actual FREE T4 training qualified; native task transfer not yet qualified

**2026-10-28. Independent owner-side audit: PASS for completed optimizer steps,
new weight integrity, and action-JSON format only.** This report is not evidence
of real application success or broad computer-use generalization.

## Frozen source/model evidence

- Private Kaggle T4 research kernel: `abdulazizshehri/hala-h164-private-original-ui-t4`. Its final job receipt reports `H164_COMPLETED_GENUINE_192_T4_OPTIMIZER_STEPS`.
- Frozen GPU training source commit: `545accc6c09db7553fa0bdbe91953c41fbed3bb5`.
- Input: **384 unique original-application TRAIN screenshots**, 192 optimizer steps with accumulation 2, exactly one image exposure per example; **96 separate original DEV screenshots**, without owner gold.
- Source ZIP SHA-256: TRAIN `ed08ef5a7e8b38d6fe8c38c235d93d8fa0625437ca3c67a5578f4fc941db4495`; DEV `17adcc78fae62d7de3ae73c2ae7d3863abd0b2f1105edf4f45db7f4c78b503b0`.
- Frozen starting H161 language LoRA SHA-256: `72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c`.
- **Resulting new H164 language LoRA** SHA-256: `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`; local bytes **21,851,336**; owner-side hash verified against Kaggle job and training receipts.
- Verified exactly **384 finite supervised loss samples**; the kernel's `optimizer_steps=192`, `train_exposures=384` and `unique_train_images=384` agree.
- Owner-only `python -m h164.audit_gpu_train --gpu-output <PRIVATE_OUTPUT_DIR>` independently checks the job/training receipts, trained adapter hash, all loss records, and the exact 96 before + 96 after model predictions without uploading gold.

## DEV format result (not task reward)

| Original application | Frozen H161 valid typed JSON | Updated H164 valid typed JSON |
| --- | ---: | ---: |
| Calculator | 48/48 | 48/48 |
| SimpleTable | 32/48 | 48/48 |
| **Combined** | **80/96** | **96/96** |

On matched DEV image identities: **16 format gains, 0 format regressions**.
These are *only* strict JSON action-schema outcomes. No original-browser click,
drag, text-action success, target localization, or OOD transfer was measured by
the training kernel. It used no owner DEV reference coordinates.

## Next independent gate (H165)

An entirely independent pre-frozen, never-TRAIN/DEV-seen OOD cohort contains
**32 Pell editor and 32 Pikaday calendar cases**. The exact original browser
source manifest was separately qualified before H164 training. The private
dataset `abdulazizshehri/hala-h164-private-native-ood-models-v1` carries
only 64 image+task inputs and the frozen H161 and H164 weights; no owner
reference points or selectors. Trained LoRA ZIP SHA-256:
`4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da`.

The frozen private FREE Kaggle T4 OOD kernel
`abdulazizshehri/hala-h164-private-native-ood-t4` will produce
**128 independent image-based model proposals (64 per model)**, *zero*
optimizer steps. Those proposals must then be run ONCE in **128 genuine
original Chromium native browser task episodes** on owner Windows and scored
by real app state, independently audited with `h164.audit_ood_native`.
The heldout cases are consumed at first model-use; never tune or retry them.
Do not promote if one OOD application fails, even when aggregate results rise.

**No H140 blind calls, paid service, public weights/media, 35B flagship update,
main merge or release. Keep both PR #154 and PR #155 Draft.**

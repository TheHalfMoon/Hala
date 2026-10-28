# H167 — independently qualified bilingual combined-LoRA feasibility on a real free T4

**2026-10-28. Research feasibility ONLY. No new real-app accuracy claim.**

## Outcome: qualified PASS

Owner-private Kaggle version 6:
`abdulazizshehri/hala-h167-private-combined-lora-t4-probe`
returned `KernelWorkerStatus.COMPLETE`. The job receipt status is
`H167_SYNTHETIC_COMBINED_FREE_T4_PASS`, and the separately downloaded
model receipt status is `H167_TWO_LANGUAGE_SYNTHETIC_STEPS_PASS`.

Independent owner Windows command:

```sh
python -m h167.audit_combined --output D:\Hala-H167-Private\sixth-run
```

Returned `H167_INDEPENDENT_COMBINED_BILINGUAL_SYNTHETIC_T4_PASS`.
The auditor validates **both independent receipts**, pinned source SHA,
external base revision, source hashes, CUDA cap, both LoRA activations,
bilingual gradient updates, and all non-leak/nonpublication claims.

## Actual evidence

- Frozen external **research** base: `patrickballou/gui-assist-grounder`,
  revision `31632edaa20749c7f532868a42635f7240033106`.
- Exact pinned executed H167 source commit:
  `b79143aa3fd1cb7fc6683ff58ed80d4dd384dfb3`.
- Previous genuine H164 trained **language adapter** rank 8, private input
  ZIP SHA `4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da`.
  Kaggle automatically unpacks uploads; the code independently
  verifies all **three** extracted files by SHA-256 before loading.
  Original trained safetensors SHA:
  `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- Both `language` and `vision` adapters were active in the **underlying
  PEFT LoRA tuner**, not just in wrapper metadata. All **96 actual
  language LoRA layers** and both `visual.merger.linear_fc1/fc2` projections
  were checked for active registration. PEFT 0.21.2's outer wrapper
  cannot accept a list of adapters; the underlying tuner can.
- Exactly **5,455,872 frozen language LoRA parameters**. A before/after
  tensor digest confirmed these weights did **not change**.
- Exactly **57,344 trainable rank-4 vision LoRA parameters**, no other
  trainable parameters. Two actual optimizer steps on the **same
  artificial 960 x 720 red-square canvas**: English step had 2 finite
  nonzero gradient tensors and 2 updated tensors; Arabic step had
  **4 finite nonzero gradient tensors and 4 updated tensors**.
- Free NVIDIA Tesla T4: peak CUDA **allocated 7.025 GiB**,
  **reserved 9.506 GiB**; allocated limit **13.0 GiB**.
  Previous H166 visual-only 10.950 GiB allocated is **not a matched
  comparison**, because H167's checkpointing/adapter setup differs;
  do not claim memory or performance improvement over H166.
- Environment: torch 2.11.0, transformers 5.17.0, peft 0.21.2,
  Pillow 11.3.0.
- Private Kaggle mount contains **only** the frozen previously trained
  adapter (README/config/safetensors). No original app images, new
  owner gold, consumed DEV/OOD or public dataset mounted. Baseline
  adapter was staged in `/kaggle/tmp`, not result outputs.
  No newly trained weights saved, published or released.
  No paid GPU, H140 blind calls, or Hala 35B changes.

## What this does NOT prove

A two-step synthetic-only optimizer update is **not** evidence that the
bilingual controller selects better click points, completes drags,
types correctly, or generalizes on any original browser application.
H164's only independently measured novel-app native reward remains
**1/64 to 10/64** (Arabic only **0/32 to 1/32**, 54 failures).
The H164 Pell/Pikaday OOD and previous H140/H162/H163 holdouts are
**consumed forever**: do not run new model tests, tune, train or select
on those screenshots. A real H168 must independently qualify new
licensed TRAIN and separate DEV original UI data, and preregister
an entirely fresh OOD browser-native reward split before inference.

**GO for new real-source TRAIN-only exploration, NOT GO for release.**
Keep PR Draft, main unchanged, no paid compute or weights publication.

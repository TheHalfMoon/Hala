# H167 frozen H164 language + small visual merger adapter

**Research feasibility only. This is not a new real-world reward measurement.**

H165's only independently measured external app reward was **1/64→10/64**
on fresh Pell/Pikaday, including Arabic 0/32→1/32. Those scenes and
other old OOD/DEV cases are consumed: never rerun them for selection,
training, tuning, or promotion. H166's independent real free-T4 *visual-only*
synthetic one-step proof used 10.950 GiB allocated, but it did NOT establish
whether two adapters compose safely or whether Arabic native success improves.

## H167 predeclared scope

- Load **the real frozen H164 trained language LoRA** from the exact private
  source ZIP SHA `4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da`;
  verify its safetensors SHA `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`,
  pinned model ID and rank-8 config. This package contains ONLY the trained
  adapter + README/config; it contains no screenshot, training gold or OOD.
- Activate both: frozen `language` adapter + rank-4 `vision` adapter
  targeting `visual.merger` only. Verify all base and language parameters
  are frozen, assert active adapters and guard **exactly four trainable visual
  LoRA tensors**. Compare actual frozen language tensor digests before/after.
- Execute exactly TWO supervised optimizer steps on the same purely artificial
  960x720 image: English and Arabic red-square-click instructions. Both
  synthetic labels are generated within source code, never from an application.
- Require free PRIVATE Kaggle Nvidia T4, exact pinned environment, actual
  nonzero finite gradients, changed visual tensors each step, optimizer
  states, language SHA before/after, measured peak allocated <=13.0 GiB.
  No paid GPU fallback, no published model/weights, no new OOD inference.
  Reject any structural or memory mismatch.
- Independent receipt audit must verify job + model evidence. No trained
  model is saved, even when the probe succeeds.

After H167, a separate H168 requires completely NEW licensed original
application TRAIN/DEV image sources and a preregistered unrelated OOD set.
Aim for Arabic click, drag, and type, and real browser-native task rewards.
Neither this synthetic step nor a valid JSON response proves quality.

**Branch stacked, Draft only. Do not merge/publish or modify Hala 35B.**

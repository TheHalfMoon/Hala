# H367 — Four-row bilingual real Qwen3-VL processor packing proof

**Research-only SOURCE/DEV review candidates; NOT legally admitted TRAIN, blind OOD, learned gain, or a releasable model.** This branch extends H366 Draft #396, which must independently pass its own exact-head GitHub checks. No main merge, no weights, no paid GPU, no model inference or adapter.

## Why a new proof is needed

H364 and H365/H366 proved only **two** actual real pinned Qwen3VLProcessor examples in the same batch (one English click + one Arabic type_text). The collator explicitly supports **2–4 source images**, but there was no actual integrated proof of 4-row processor packing and all EN/AR × click/type_text combinations.

This new opt-in real owner Windows CPU script sources four original H363 ControlGarden / H358 NoteRelay candidate images from actual offline Edge browser, checks their original native action/source audit and routes each through the exact pinned public Qwen3VLProcessor. It independently proves **four-row** image patch concatenation, grid/label/text association, right-padding and assistant-only masking; then **reverses all four rows** and proves exact output order and aggregate invariance again.

The exact fixed public review-candidate selection (not concealed or new OOD): ControlGarden 0 EN click, ControlGarden 1 AR click, NoteRelay phase-type index 2 EN type_text, NoteRelay phase-type index 6 AR type_text. All 192 original candidate SHA/action rows are independently audited before source selection. Source text/target, images, model tokens and gold remain ephemeral; only aggregate processor counts and refusal/safety flags are printed. Pinned eight public processor files are SHA-verified before Transformers import; offline flags and CPU-only torch must pass.

The GitHub workflow tests the **source-selector and stdlib H364/H199 contracts** across Windows/Ubuntu Python3.11/3.12, **not the real Qwen processor**; real pinned processor must be run on authorized owner Windows and evidenced separately. Both are mandatory for H367 qualification.

## Actual owner Windows real CPU evidence (2026-11-02)

- Owner offline Qwen3VLProcessor exact eight public H218 SHA-pinned processor files; **real original Edge browser source capture**, **192/192 original synthetic candidate image/action SHA identities audited**; no weights or optimizer.
- Four real source rows: EN click 1, AR click 1, EN type_text 1, AR type_text 1. The normal and **reversed** full 4-row visual tensor/token/assistant-only right-padding/attention/grid association both passed exactly.
- Aggregate: **4 image grids, 10,560 visual patch rows, 78 supervised assistant-only tokens, 23 masked PAD tokens**; returned `normal_and_reverse_order_exact=true`, `model_weights_loaded=false`, `model_forward_calls=0`, `optimizer_steps=0`, `can_train=false`, `can_release=false`.
- Owner Windows Python3.11 and Python3.12 **23/23** stdlib source selection/native mask/overflow tests PASS; syntax compilation/diff checks PASS. This **does not qualify H367 GitHub exact-head CI**, which must be checked on the PR independently.
- Original source PNGs and native gold were generated under a TemporaryDirectory and deleted before the aggregate result. No private screenshots, credentials or model weights were transferred or committed.

## Blocking boundaries

H199 approved training manifest allowlist remains empty; H355 independent rights keys empty; H356 independent OOD custodian keys empty. Rights reviewer [#384](https://github.com/TheHalfMoon/Hala/issues/384) and fresh sealed source-disjoint EN/AR OOD custodian [#385](https://github.com/TheHalfMoon/Hala/issues/385) are unresolved. This is NOT a 4-sample model quality score; there is no model forward/generation or optimizer and no new learned accuracy. No merge or release.

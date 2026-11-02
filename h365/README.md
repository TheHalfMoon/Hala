# H365 — Integrated H363 + H364 original-source multimodal batch

**Source-only research. NOT authorized TRAIN, model learning, blind OOD, merge to main, or release.**

H363 [#393](https://github.com/TheHalfMoon/Hala/pull/393) and H364 [#394](https://github.com/TheHalfMoon/Hala/pull/394) are **sibling** PRs on H362. Separate branch CI results cannot qualify their combined code. H365 merges exact H363 `70a76adb6c6775a4c0c96db246c99a8b2eb03625` with exact H364 `8d60413f9ce8640e0d5e01e99774df49f4f74eee` on an isolated worktree; the two source branches and `main` remain untouched.

H363 removes a documented language-independent target-parity shortcut from original ControlGarden visual examples. H364 builds variable-length, 2–4 source-image Qwen3-VL **processor-only** CPU batches with proper assistant-only training labels, image/user/padding tokens labeled `-100`, correct attention masks, per-source image grids and visual patch concatenation in row order. Neither is official TRAIN data or a model optimizer. H365 adds a **new fail-closed check**: each real `image_grid_thw` volume T×H×W must exactly equal its flattened `pixel_values` patch count and have integer dtype. The isolated real CPU run also attempts one forged grid and one truncated-patch input and must reject both.

On the integrated H363 source, the owner Windows real CPU runner generates original ControlGarden/NoteRelay scenes in temporary directories, verifies all 192 public synthetic candidate PNG/action SHA256 identities using H359, and independently creates two real Qwen3VLProcessor examples: one English native click and one Arabic native `type_text`. It verifies exact H199 assistant-only completion, image tensors, multimodal token prefix and padded tensor geometry. All source PNGs and gold are deleted before outputting counts. Uses only the pinned H218 SHA-verified eight PUBLIC processor assets at revision `89644892e4d85e24eaac8bacfd4f463576704203`, entirely offline. **No model weights, gradient, forward inference, optimization, GPU or paid service**.

## Actual integrated owner Windows CPU result (2026-11-02)

After merging the source changes, the REAL pinned offline
`Qwen3VLProcessor` (not simulated) completed the combined source
and batch audit twice, including the new negative guards. Final
`H364_REAL_CPU_NATIVE_BATCH_ONLY` aggregate receipt:
`all_192_source_rows_audited=true`, `image_grids=2`,
`visual_patches=5280`, `languages=en1/ar1`,
`actions=click1/type_text1`, `assistant_supervised_tokens=39`,
`padded_tokens=12`, `grid_patch_mismatch_refused=true`,
`truncated_patch_mismatch_refused=true`,
`batch_tensor_dimensions_valid=true`,
`model_weights_loaded=false`, `model_forward_calls=0`,
`optimizer_steps=0`, `official_train_admitted=false`,
`can_train=false`, `can_release=false`. Source screenshots and gold
were ephemeral and removed before producing the aggregate receipt.
This is new true processor input parity on H363 visuals; it does
NOT measure learned Qwen weights, performance or generalization.

The dedicated GitHub workflow replays the original source browsers and H363 parity audit together with H364 assistant-mask tests on Windows Edge / Ubuntu Chromium Python 3.11 and 3.12; it does **not** run the actual Qwen model or real pinned processor. Its checks are source and pure-Python contract evidence only.

H199 approved TRAIN digests remain EMPTY, H355 independent rendered-image ML rights reviewer registry EMPTY, H356 new blind source-disjoint EN/AR OOD custodian registry EMPTY. Legal issue #384 and independent benchmark custody issue #385 remain unresolved. Hala remains **NO-GO** for TRAIN and release until independently lawful source/admission, truly unseen OOD, and real trained-vs-pinned-base improvement and safety.

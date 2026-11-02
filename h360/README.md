# H360 — original ControlGarden bilingual target-position confound repair

This is a **source-only, unadmitted** engineering correction to H357, stacked on H359. It does not create a TRAIN release, use a model, score private OOD, or authorize screenshot ML rights.

## Observed defect

H357 previously used `chosen = index % 4` while `language = "ar" if index % 2 else "en"`. Thus all 32 English candidate targets occupied only the two even slots, while all 32 Arabic targets occupied only the two odd slots. Language alone disclosed half of the possible target positions; model performance could be inflated by this source-design shortcut. These source-exposed candidate rows were never independent blind OOD.

## Bounded repair and checks

`chosen = (index // 2) % 4` rotates the selected slot by bilingual pair. For each language separately, 32 original scenes now distribute the four selected slots **8/8/8/8** and the four selected colors **8/8/8/8**. The source HTML blob, four-button GUI, source-family identity, source-only split and H199 no-admission policy are unchanged. Existing source-browser tests replay actual native clicks and require correct original browser state and 64 distinct screenshots. H359 separately audits generated native H199 messages, screenshot bytes, source provenance and legacy schema rejection.

This fixes **only slot and color marginal confounding**. Shape frequencies and source-family diversity are not guaranteed balanced. The scenes remain one exposed original synthetic source family, not an independently licensed TRAIN/DEV release or fresh blind OOD. No pretrained/adapted Qwen comparison or learned improvement is claimed. H199 `ADMITTED_TRAIN_MANIFEST_SHAS` and H355/H356 trusted external registries remain empty; independent legal rights and blind source-disjoint custody are still blocking.

All generated image/label rows remain ephemeral in owner-only test temporary directories; no media, prompts, gold, weights, credentials or private evaluation data are committed.

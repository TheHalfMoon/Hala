# H170 — freeze and diagnose whether H168 vision LoRA affects actual forward logits

**Reason:** independent H169 real browser DEV returned **5/32 BEFORE and 5/32 AFTER** with **0 paired wins, 0 paired regressions**, and separate post-hoc diagnostics confirmed **0/32 raw greedy responses changed at all** after activating the H168 visual adapter. H169 DEV is consumed forever. Do NOT retrain, test, or select on it.

Before any additional costly training or new OOD, test whether the H168 rank-4 visual-merger LoRA actually changes the model's visual activations and final-token logits when composed with H164 frozen language LoRA. If the numeric effect is nonzero but greedy actions unchanged, the issue is likely insufficient margin/learning or restricted model capacity, not simply disabled LoRA. If zero in actual forward, isolate architectural routing/PEFT activation before more training. Neither finding proves a new browser quality gain.

## Strict experiment preregistration

- Use exactly **ONE known existing H168 TRAIN source image**, specifically `choices` case 168000 from the pinned original H168 32-TRAIN ZIP (SHA `d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6`).
- **ZERO** DEV or OOD data, old screenshots, owner-heldout gold or benchmark model calls.
- Compare the exact same 960×720 image and visible instruction under (A) active H164 **language only** and (B) active H164 **language plus H168 vision**.
- Frozen base `patrickballou/gui-assist-grounder` at `31632edaa20749c7f532868a42635f7240033106`.
- H164 frozen language weights SHA `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- H168 private vision weights SHA `506f571d0ff45bbe8932d9ea1f56dd6e4b5b5dc5f90cfdd1397243285523cd88`.
- Real Tesla T4 free Kaggle; torch/transformers/PEFT pinned; no paid GPU, no training step, no weight saved. Peak allocated cap **13 GiB**.
- Capture from actual forward hook on `visual.merger.linear_fc2`: finite numeric tensor difference and number of nonzero elements. Also compute max absolute, L2 and nonzero elements of the real full-vocabulary final-token logits.
- The input prompt must include **zero training supervision** even though the source belongs to TRAIN. Test is diagnostic only; not new native browser reward, and does not consume any DEV/OOD holdout.

## Next engineering decision

1. **If both vision and logit deltas are exactly zero:** reject H168 adapter as forward-ineffective. Diagnose module targeting, PEFT routing and model architecture before training.
2. **If deltas are nonzero but greedy DEV actions stayed identical:** H168 adapter influences forward but the 32 steps were insufficient to change action decisions on evaluated tasks. Plan broader new original TRAIN with click/type/drag and carefully controlled stronger visual adaptation; new DEV and OOD must remain never seen until one-shot.
3. No model release/promotion unless true new independent native app reward supports it.

The separately staged original MIT flatpickr and Awesomplete sources under owner D were successfully smoke-proven for genuine calendar and autocomplete click actions. Their candidate OOD/novel-app source images have **not been captured or evaluated**. Keep them unused while diagnosing H168, so they remain eligible for future genuinely unseen-app evaluation.

No main merge; no public weights; no Hala 35B change; Draft PR only.

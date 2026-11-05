# H419 authenticated source-original model error attribution (H425 design input)

Source: owner PRIVATE original H419 completed Kaggle receipt, SHA256 f8c491d8522a8349974d05318fded7cbe0349478be27efcbb118a8efdcfa37ac. Owner downloaded actual LoRA SHA256 96d07b07cca54d15e9ce11da4d26fa3748564f09a2974d9a0b48e419ab03eb8c. All 240 source-original DEV images and case identities independently replayed on Windows; raw original source images, predictions, case IDs and labels remain PRIVATE and are NOT committed to GitHub. The aggregate table below is computed directly from original parent_dev_exact and adapted_dev_exact booleans and cohort position. **These are author-visible source development results, not independent sealed OOD.**

## Aggregate genuine paired errors (old trained H417 → genuine full-run H419)

| Original DEV cohort | n | H417 correct | H419 correct | Newly correct | Newly incorrect | H419 valid |
|---|---:|---:|---:|---:|---:|---:|
| OLD H417 source DEV | 144 | 76 | 71 | 7 | 12 | 144 |
| NEW H419 source DEV | 96 | 18 | 30 | 17 | 5 | 96 |
| ALL source DEV | 240 | 94 | 101 | 24 | 17 | 240 |

Source-only research model gate required OLD >=72 and NEW >18 with 240 valid; original full H419 final FAILS because OLD=71 despite total 101. Do not promote H419 final or train H422 from its failed-source-gate final adapter.

## Original OLD H417 source DEV error attribution

| True source domain | old n | H417 correct | H419 correct | Newly correct | Newly incorrect | Net change |
|---|---:|---:|---:|---:|---:|---:|
| sheet | 34 | 17 | 14 | 1 | 4 | -3 |
| slide_resize | 34 | 11 | 11 | 0 | 0 | 0 |
| slider | 36 | 27 | 26 | 3 | 4 | -1 |
| text_highlight | 40 | 21 | 20 | 3 | 4 | -1 |
| Total | 144 | 76 | 71 | 7 | 12 | -5 |

NEW H419 source DEV (no sheet): slide_resize 0→3/32, slider 11→16/32, text_highlight 7→11/32. Old H417 Arabic original24 went from24→23 correct, so preserving genuine Arabic as well as English is a distinct requirement.

## Engineering implication for H425

Original H419's 37,808 gradient-bearing passes used **30,000 NEW original source JPEGs** and 976 Arabic browser screenshots replayed four passes; it did not replay the original 10,976 old H417 English TRAIN screenshots. The old DEV regression is concentrated in the original previously mastered source family **sheet** while new source families improved.

H425 is a separate model-training hypothesis rather than a claim of improvement: train the **previous genuine H417 LoRA parent SHA** directly with 16,000 original source screenshots each carrying TWO different genuine native drag instructions (32,000 action targets) PLUS 10,976 original H417 English TRAIN images and 976 genuine original Arabic browser TRAIN images replayed eight passes. The H425 frozen plan has 27,952 distinct train images and 50,784 prospective image-conditioned gradient passes / 12,696 optimizer steps; original H417 OLD144 and H419 NEW96 source DEV plus reserved H421 192 unseen screenshot counterfactuals are excluded from gradients.

After actual GPU run, compare case-level old sheet/slider/text_highlight nonregression; require original frozen source gate old144>=72, fresh96>18, 240 valid, PLUS total and Arabic safeguard. The new old-sheet focused error breakdown motivates more detailed verification but does NOT prove H425 will fix forgetting. Always keep separate genuinely sealed bilingual external OOD/rights/reviewer release gates.

No model training, sealed OOD, rights review, independent review, public weights, paid compute, or main merge is authorized by this analysis alone.

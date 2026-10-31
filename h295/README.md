# H295 — separate *descriptive* paired GUI research from evidence of model improvement

## Why this gate exists

Two genuine owner-private FREE T4 Qwen3-VL-2B experiments completed with real
LoRA training and real native GUI post-action source-owned state scoring:

| Verified first-party research | Pretrained BASE | ADAPTED | Measured difference |
|---|---:|---:|---:|
| H291, original Queueboard, 8 LoRA steps | 1/2 | 1/2 | 0 |
| H292, original Queueboard, 8 LoRA steps | 1/2 | 1/2 | 0 |
| H292, separate original RouteBoard, 8 LoRA steps | 5/6 | 5/6 | 0 |

Real H292 combined *exposed research* descriptions are **BASE 6/8 vs
ADAPTED 6/8**. The figures are independent of raw images, model text,
action coordinates and weights and were previously validated by strict
owner-private H292/H288 receipts. **H293's separate private 24-step
study ended in ERROR with NO readable allowlisted receipt.** Its
optimizer count, model score and failure stage remain UNKNOWN. Do not
turn its error into 0/8 or any claim about 24-step adaptation.

Public source-owned Queueboard and RouteBoard are already exposed and
evaluated. Their sample sizes are tiny. No new OOD, official TRAIN/DEV
admission, generalizable skill gain, release or model weights are established.

## What H295 adds

`h295/paired_score_integrity.py` is a **pure Python, zero-cloud,
CPU-only** statistical validity check. Its accepted input contains ONLY
already sanitized grouped case counts. No raw Kaggle artifacts,
screenshots, task text, case IDs, gold, model outputs or coordinates
are required or accepted.

For each *fixed original publicly exposed* family:
- Queueboard has exactly **2** original research cases.
- RouteBoard has exactly **6** original research cases.
- `base_success` / `adapted_success` are exact integers, not
  bools or invented per-scene rows.
- Optional `paired` must supply the complete compatible 2×2
  outcomes: `both_success`, `base_only`, `adapted_only`,
  `both_fail`. Incorrect marginal reconciliation is refused.
- When true paired counts are present, compute **exact conditional
  two-sided McNemar p** using only discordant pairs (binomial tail),
  not an inappropriate unpaired-proportion z-test.
- If only marginal totals are available and unequal, the paired p-value
  is **not identifiable**; it is returned as `null`, not guessed.
- If the marginal totals are equal, `base_only == adapted_only` for
  *every possible* pairing, so the exact two-sided conditional
  McNemar p-value is **1.0** even without private row-level outcomes.
  This is a mathematical consequence, not a new neural evaluation.

For H292's previously published sanitized counts, both research
families have p=1.0 and delta=0, and neither supports a claim that
the adapter improved GUI task success. A more favorable experimental
delta in future would *still* require an independent, unused blind
holdout and external rights/source review.

**No p-values are pooled across families**: source dependence and
unknown pairing/experimental selection would make this misleading.
The audit always reports `official_train_dev_admitted=false`,
`new_independent_blind_ood_proven=false`,
`statistically_validated_generalization=false` and
`can_publish_or_release=false`.

## Small public-research example only

The following is **not** an owner-private Kaggle record; it is merely
the already public high-level H292 counts, with no per-scene outcomes:

```json
{
  "schema": "H295_EXPOSED_PAIRED_NATIVE_RESEARCH_V1",
  "source_exposure": "public_previously_exposed",
  "groups": [
    {
      "family": "queueboard", "sample_count": 2,
      "base_success": 1, "adapted_success": 1, "paired": null
    },
    {
      "family": "routeboard", "sample_count": 6,
      "base_success": 5, "adapted_success": 5, "paired": null
    }
  ]
}
```

To use the pure local reader, put only non-sensitive public research
counts into a file ending with `.h295-public-score.json`:

```powershell
python -m h295.paired_score_integrity --aggregate "D:\<your-public-score>.h295-public-score.json"
python -m unittest discover -s tests -p test_h295_paired_score_integrity.py -v
python -O -m unittest discover -s tests -p test_h295_paired_score_integrity.py -v
python -m compileall -q h295
```

Its schema is intentionally closed; unknown fields, fake blindness,
mismatched protocol sizes, malformed/duplicate escaped JSON keys,
non-finite numbers, symlinks, fake booleans and made-up 2×2 pairing
are refused. Input authenticity is **NOT** verified by this reader.
The H288/H292 private kernel and closed receipt verifier must
independently authenticate genuine results **before** a public
sanitized aggregate is derived. Never feed H295 raw private reports.

## Release barriers remain

H199's formally admitted TRAIN manifest is EMPTY. H173 old scored
OOD is consumed/sealed; H212 scored tasks cannot be reused; H250/H251
third-party media remain quarantined. A defensible final Hala
qualification still requires independently documented rights and
train/DEV admission, diverse source families and genuine
non-exposed independent blind OOD, with real paired adapted
native task-state improvement, reliability and abstention.
Everything remains in stacked DRAFT PRs; no merge/deploy/release,
no paid inference, no Mac.

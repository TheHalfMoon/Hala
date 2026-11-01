# H310 — prospective paired native-GUI evaluation design, without private outcomes

**SOURCE-ONLY STATISTICAL ENGINEERING. NOT AN OOD SCORE, A NEW HOLDOUT,
TRAIN/DEV ADMISSION, A Qwen run, OR EVIDENCE OF MODEL IMPROVEMENT.**

## Scientific gap

H295 correctly computed exact conditional two-sided McNemar significance for
the previously exposed H292 paired research (BASE 6/8 vs ADAPTED 6/8,
no gain). It did **not** provide a prospective, auditable design tool for
determining whether a *future, independently custodied* paired evaluation
could have enough discordant cases to detect a hypothetical benefit.

H310 provides exact rational arithmetic for **discordant pairs only**.
For `m = base_only + adapted_only`, the two-sided conditional McNemar
p-value is `min(1, 2 * P[Binomial(m, 1/2) <= min(base_only, adapted_only)])`.
For a *hypothetical* probability `q` that ADAPTED wins among discordant
pairs, conditional positive-direction power is the exact probability of
`adapted_only > base_only` **and** two-sided p <= alpha under
`adapted_only ~ Binomial(m, q)`.

This does not estimate `q` from H292, private model outputs or a sealed
benchmark. The default q=3/4 is **illustrative**, not an observed effect.
Alpha=1/20, target conditional power=4/5 and search m=1..128 are explicit
planning choices, not preregistered evaluation decisions.

## Exact illustrative result

For q=3/4, alpha=0.05 and target conditional power=0.80:

- With **8 discordant pairs**, conditional power is exactly
  `6561/65536` (approximately 10.01%). This is not a claim that H292
  contained 8 discordant pairs; it had **8 total exposed tasks** and no
  authenticated public per-task pairing.
- The **first** discordant count meeting 80% is **30**.
- Power can dip below 80% at 31 and 32 because the exact test is discrete.
  Thus 30 is not a monotonic sample-size guarantee.
- Within the **finite tested range m=1..128**, all counts from **35**
  onward meet 80%. This is NOT a guarantee for m>128.
- **Number of GUI tasks required remains UNKNOWN**. It depends on the
  unknown rate of discordance and source-family dependence. Do not turn
  the discordant count into a total-task count or claim independent trials.

## Reproduce, no GPU or private input

```bash
python -m h310.exact_discordant_study_design
python -m unittest discover -s tests -p test_h310_exact_discordant_study_design.py -v
python -O -m unittest discover -s tests -p test_h310_exact_discordant_study_design.py -v
python -m compileall -q h310
```

No input files, model calls, weights, source screenshots, task text, labels,
gold, predictions, points, raw receipts, historical case identifiers, or
third-party media are read, stored or emitted. The code accepts only strict
`Fraction` hypothetical probabilities and integer discordant counts; no
private case-level or aggregate reader exists.

## Non-negotiable gates outside this calculation

A valid future scientific result still requires independent legal rights and
source lineage review; officially admitted source-diverse TRAIN/DEV (H199
allowlist remains EMPTY); a *new*, never-exposed, separately custodied
source-disjoint blind OOD set; prospectively frozen paired evaluation,
source-family-aware uncertainty, abstention and safety gates, and genuine
model-time BASE-vs-ADAPTED native task-state gains.

The consumed H173 scored OOD remains SEALED, H212 scored cases barred,
H250/H251 third-party media quarantined, H293 private error output access
safety-blocked, and H294 provenance write safety-blocked. This code does
not touch or work around any of them. All H309/H310 sources remain public
research only; no release/merge/model-weight claim is supported.

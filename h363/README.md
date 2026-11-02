# H363 — break the H357 task-blind target parity shortcut

Status: **source-only candidate engineering; NOT admitted TRAIN, DEV, blind OOD, or release**.

## Observed scientific defect

The H362 ControlGarden generator balanced target slot, shape and color
marginals, plus each pair of those factors, within both English and Arabic.
However, its target color index used `slot + shape_index + 2*repeat` modulo
four. This made `(slot + shape_index + color_index) % 2 == 0` true for
**all 64 targets**. A task-blind attacker that selected uniformly among
visible parity-even tiles had **28/64 = 43.75% theoretical expected accuracy**,
versus 25% uninformed chance. This is an arithmetic source audit, not Qwen.

## Bounded source fix and adversarial regression

The color index now uses `slot + shape_index + repeat` modulo four. The
two repetitions of each (slot, shape) pair receive different parity. Each
language retains 32 cases; all 16 combinations for each of slot x shape,
slot x color and shape x color occur twice, with 8 examples per marginal.
All 64 original source layouts remain distinct, and no external image assets
or original HTML are changed.

The exact same **instruction-blind** even-parity attack now scores
**63/4 = 15.75 expected correct of 64 = 24.609375%**. This expectation
averages uniformly over eligible visible tiles, falling back to all four
when no even-parity tile exists. The negative control verifies the old
formula would make all 64 target parities even. The tests verify 16 even
and 16 odd targets **within each language**, plus every pairwise balance.

This removes one measured label shortcut. It **does not prove the absence
of other shortcuts**, independent statistical generalization, real Qwen
performance, or any training gain. Case-ID or generation-order leakage,
higher-order associations, and model-dependent exploitation remain possible.

## Boundaries

- No GPU, model weights, processor inference, optimizer, paid service or
  private evaluation artifacts are used by these tests.
- H357/H358 images and gold remain temporary original-source candidates.
- H199 official TRAIN allowlist and independent reviewer/custodian registries
  remain empty. No rights/title/derivative-weight permission is inferred.
- H173 is consumed, H212 barred, H250/H251 quarantined, H293/H294 restricted.
- Independent authorized source-family-disjoint bilingual blind OOD and
  matched BASE-vs-ADAPTED Qwen evidence are still required.

Run: `python -m unittest tests.test_h363_target_parity tests.test_h362_source_balance -q`.

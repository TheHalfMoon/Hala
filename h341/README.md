# H341 — Prospective family-level statistical power bounds (no model evidence)

**Status: invented-probability design mathematics ONLY. No model inference, training, rights grant, admitted data, private receipt, independent blind OOD, or EGM Goldfish approval.**

H340 uses a two-sided sign test over independent GUI **source families**, with Holm correction for two preregistered English and Arabic RTL endpoints. RailKit H325 and SignalDesk H338 supply only **two publicly exposed, rights-unapproved TRAIN candidate families** (144 scenes, 288 action-stage candidate messages). H199 has admitted no family. Hundreds of synthetic permutations cannot substitute for independent families.

H341 calculates exact marginal probability of a positive family sign-test result under *hypothetical* independent-family win/loss/tie probabilities. It then computes conservative **Fréchet lower and upper bounds** on the probability that both language endpoints pass Holm, because EN/AR dependence is unknown. It does **not** multiply the marginal powers or assume languages are independent.

## Feasibility finding

Even if every family favors ADAPTED, **six independent families cannot pass both Holm-adjusted two-sided endpoints at 0.05**. **Seven all-winning independent families are the mathematical minimum**. This is **not** a sample-size recommendation or a power guarantee. Ties, losses, lower effect sizes, and correlations reduce power. A dataset with 288 action samples from two families cannot fix this inferential limitation.

The assumptions are explicit: source families independent of one another; one win/loss/tie sign per family per language; fixed hypothetical conditional adapted-win probability given a non-tie; fixed hypothetical tie probability; and no post hoc family selection. H341 does not establish any of these assumptions for real model evaluations.

Example with **invented probabilities only**:

    from fractions import Fraction
    from h341.familywise_design_sensitivity import design_sensitivity
    hypothetical = design_sensitivity(
        16, Fraction(4, 5), Fraction(3, 4), Fraction(1, 10)
    )

The output is exact rational **hypothetical design sensitivity**, not an observed Qwen result or confidence interval. Do not put private evaluation aggregates, source gold, model outputs, predicted coordinates or receipts in GitHub.

## Acceptance and next decision

Run invented-only CPU tests:

    python -m unittest discover -s tests -p test_h341_familywise_design_sensitivity.py -v

The tests cover exact six-versus-seven thresholds, ties, null-tail conservatism, unknown-language-dependence bounds, invalid inputs and false scientific-qualification flags. Free GitHub Actions checks run on Ubuntu/Windows with Python 3.11/3.12 and no GPU or datasets.

**Actual next gate:** independent copyright/title and rendered-screenshot ML training rights for RailKit and SignalDesk; separately authorized new source-family-disjoint DEV and never-exposed private blind OOD; genuinely adapted Qwen trained only on H199-admitted lawful data; then preregistered paired native image-conditioned GUI evaluation with safety and abstention. H173 remains sealed; H293/H294 safety blocks remain. Google EGM Goldfish fresh-context review has not been performed (REVIEW_BLOCKED). No train, release, or main-branch edit is authorized by H341.

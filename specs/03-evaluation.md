# H3 Synthetic evaluation and H7 data factory

## Goal
Execute synthetic tasks and verify independent final state; produce frozen bilingual and GUI suites with provenance and split isolation.

## Coordination
Depends on: H2
Surface: evaluation, data, tests/test_evaluation.py
Review: full

## Requirements
Separate proposal provider, observation, validator, executor, postcondition checker, metrics and serializer. At least 60 bilingual intent cases and 120 GUI cases, family identifiers, train/dev/test isolation, Arabic/RTL and English, DPI variations, similar controls, stale observations, drag-and-drop and bounded recovery workflows. Model-independent baselines are infrastructure qualification only.

## Acceptance criteria
- Frozen deterministic manifests reproduce byte-for-byte and prevent family leakage and duplication.
- Executor changes synthetic state; checker derives success from state independently of model claims.
- Negative controls reject wrong targets, stale proposals and false success; E3 multistep workflows enforce budgets.
- Provenance and rights validator rejects unknown and benchmark-evaluation sources for training.
- Metrics include sample counts, invalid outputs, wrong-target and verified success, bilingual strata and uncertainty.

## Hardstop
No real model claims, test-to-train leakage, personal data, or kernel execution.

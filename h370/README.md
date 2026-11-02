# H370 — causal, privacy-safe original-browser CI failure diagnostics

**NO MODEL TRAINING, BLIND OOD, learned gain, private screenshot exposure or release.** This is an instrumented research-only browser test qualification, stacked on H369 Draft #399. Retains source-browser strict fail-closed behavior.

## Independently verified failures requiring causal investigation

- **H368 Draft #398:** exact HEAD `da32d09d0ba9f0c702a78ad56d199d7f87b77045`; **55/56** GitHub checks success, inherited **H319** Ubuntu Chromium original bilingual dynamic `beacon_11` raises generic `H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED`, [run 41509879527 / job 125498663569](https://github.com/TheHalfMoon/Hala/actions/runs/41509879527/job/125498663569).
- **H369 Draft #399:** exact HEAD `4940d854f40fe44e2ffe138bc1ceb849f8a955b9`; **67/68** checks success, inherited **H337** Ubuntu Chromium `test_original_12_source_owned_centers_only` raises generic `H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED`, [run 41511236311 / job 125503303261](https://github.com/TheHalfMoon/Hala/actions/runs/41511236311/job/125503303261). All new H369-only 4/4 source checks passed.

These are different inherited browser tests, NOT evidence of Qwen model learning or that H369's token guard is wrong. Both handlers originally hid **every underlying exception** by raising a generic refusal without any diagnostic stage. A CI rerun without evidence could conceal a real flaky safety bug.

## H370 change

- Retain the original fail-closed `ValueError` refusal prefix, strict source hashes, native action/DOM isolation, rendered screenshot integrity and permissions.
- For ORIGINAL, PUBLIC synthetic source GUI tests only: emit a fixed **stage enum** and a tiny allowlisted **exception category** on error; never include exception message, stack trace, point, DOM data, screenshot bytes, task, synthetic gold, path, predicted action, model tokens or private data. Unknown Python exceptions map to `other`.
- H319 stages: initial scene, first image/policy, native click, post-click source, second image/policy, native type, final source.
- H337 stages: initial scene, first image/policy, native click, post-click source, optional miss geometry and final URL.
- Invented-only tests deliberately inject a secret-containing exception into both flows and prove the formatted refusal has only the bounded stage/kind values. New 4-way source-only GitHub workflow Windows+Ubuntu x Python3.11/3.12. Existing inherited H319 and H337 opt-in CI workflows still run **real original browsers**, separate from source-only tests.

## Qualification policy

**Instrumenting does not by itself repair the root cause.** Observe any future failing stage/type from an exact-head CI or repeated controlled browser run and fix ONLY an evidenced source/evaluator root cause with adversarial regression, never loosen the fail-closed safety assertions, silently rerun for green, or relabel error as a legitimate miss. The H199 official TRAIN allowlist, H355 independent rights keys and H356 independent custodian keys are empty; [rights #384](https://github.com/TheHalfMoon/Hala/issues/384) and [fresh source-disjoint EN/AR OOD #385](https://github.com/TheHalfMoon/Hala/issues/385) remain external blockers. No merge or model release.

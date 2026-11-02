# H371 — Repeated original-browser source-path evidence, not model quality

**Research-only, no ML TRAIN, model weights, novel blind OOD, or release.** Stacked on qualified H370 Draft PR #400 exact head `d875ea97b84a0db0780e7ba516ce56d6babf7ca3`. No main merge.

## Why this is required

Prior exact-head Ubuntu Chromium inherited browser CI results genuinely failed despite source-only CPU safety improvements:

- H368 PR #398 55/56 SUCCESS; H319 original 12-scene bilingual two-step source run failed on exposed synthetic scene `beacon_11`; original underlying exception masked by legacy general refusal
- H369 PR #399 67/68 SUCCESS; H337 original 12-scene first-click-center run failed similarly with a generic refusal
- H370 PR #400 56/56 SUCCESS **after adding strictly redacted stage/type diagnostics**, including 4/4 new diagnostic source-only workflow; this green execution is **not proof** that past intermittent browser failures are repaired

The next test must try to actually reproduce the original dynamic GUI failures under the same H319/H337 REAL original-source browser workflows. Rerunning PR checks without a bounded protocol obscures the flake rate and cannot prove cause.

## Controlled bounded experiment

- `run_browser_repeats.py` invokes the **existing full H319 and H337 browser unittest modules** in separate child processes for each of `--repeats 1..4` trials. Source-owned original image/state/evaluator validation, network-blocking browser context and fail-closed actions unchanged. Each H319 suite must run exactly 9 checks and H337 exactly 7: **16 real browser tests per complete trial**.
- Test stdout and stderr remain captured in memory and are **never printed**, even upon failure. Only experiment round, flow (`h319` or `h337`), exact test count on pass and H370's bounded allowlisted stage+kind on failure are shown. Unknown exceptions show `unclassified`; no scene IDs, source tasks, screenshots, raw coordinates, gold, traceback or raw exception messages leave the subprocess.
- Hard four-repeat cap, per-suite 240-second timeout; any timeout, unexpected count, nonzero result or unclassified failure is a FAIL, not an omission. There is no automatic retry within a trial to suppress red signals.
- New GitHub workflow executes **3 consecutive full trials** each under Ubuntu 24.04 pinned Chromium and Windows pinned original Edge source tests; source-only redaction tests run before browser trials. Both jobs use pinned Playwright 1.58.0 and Pillow 11.3.0; no free or paid model inference.
- CLI scripts are research-only: never convert results into fresh sealed blind EN/AR OOD, nor claim learned Qwen improvements. A 3-trial PASS shows only no failure **in that bounded sample**, not that historical browser intermittence is fixed. Any redacted failure is a starting point for a root-caused targeted fix, never a reason to weaken the evaluator.

## Hard unresolved Hala gates

The H199 officially admitted TRAIN SHA allowlist, H355 authorized independent legal-reviewer keys, and H356 independent sealed blind OOD custodian keys remain empty. Independent rendered screenshot ML / derivative weight rights [Issue #384](https://github.com/TheHalfMoon/Hala/issues/384) and genuinely NEW never-exposed source-family-disjoint bilingual blind EN/AR OOD [Issue #385](https://github.com/TheHalfMoon/Hala/issues/385) are outstanding. No lawful adapted model gain has been measured. No merge to main, training or release.

# H17 independent runtime-scope delta

**PASS for the prospective documentation change.** The docs now require a fresh model subprocess per mode while allowing reuse of the existing verified free Linux Colab T4 allocation only after preceding model processes exit. The spec requires disclosure of allocation reuse and timing confounding. This preserves separate model lifetimes without requiring a new GPU allocation. Actual int8/NF4 results remain pending, and the change does not claim qualification.

Verified root checkout `work/Hala` with the real source verifier against freeze SHA256 `e35342f0e6f58b53c25ee7220ca5ee25484673d718096791e7680823ff975d46`. Compared its file map with the previously reviewed preparation checkout: all eight frozen files other than doc/spec are unchanged, including runner, helper, config, baseline, official sources, tests and freeze/capture tools. All ten H14 dependencies verify unchanged. Freeze declares zero model runs and `before_quantized_model_runs`.

New doc SHA256: `ea60e5691137c1f3e5d51eff0d8c52caab8ad50b4de9305ca3b70e9f2191bd39`.
New spec SHA256: `866a15d17f22bb6637bb40e591b71e244b2c36b5964be890a79d86ae76bc56b1`.

- Must-fix: none in this delta.
- Worth-considering: when collecting evidence, record process isolation, pre-load device memory and allocation reuse explicitly; the runner already records pre-load allocator/device metrics. Model workers must exit before the next mode starts, as the docs require.
- Skip-it: new GPU provisioning or a full CPU-suite rerun is unnecessary for this doc/spec-only change; unchanged code retains the prior independent 136-test pass.

This review does not authorize paid compute, weights on the Windows PC, tuning, accuracy-equivalence/adoption claims, merge or production release. Publish this prospective freeze before the first quantized model run. No model execution or source mutation was performed by this judge.

panel: light combined correctness/parsimony/product delta (inherited session model); separate security and mechanical gates remain the caller's responsibility.

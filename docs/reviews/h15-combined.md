# H15 independent combined review

Scope: immutable `../h15-grain.diff`, SHA256 `b8aea10edd4a80d74784093ea075cd67802db42fbab2bd5b65790a224ee496b8`; five files in `../h15-grain.json`, with every inventory digest independently matched. Context: PRODUCT.md, CLAUDE.md, H15 spec, and existing trainer/checkpoint contracts. No source edits, weights, remote GPU execution, commits, or pushes. Security/deserialization is assigned to a separate judge.

## Correctness

- **Must-fix:** `training/scripts/qualify_lora.py:103-111` — comparison excludes checkpoint loss metrics; qualification can pass despite a loss-trajectory discrepancy whenever final adapter/optimizer/RNG comparisons pass. The requested comparison includes metrics. Add a component comparing ordered `step`/`loss` values at declared tolerances and a mismatch test; keep elapsed seconds and memory measurements as observed evidence, since those cannot reasonably require equality.
- **Worth-considering:** `tests/test_lora_qualification.py:19-30` — the interruption test calls a fake publish and then the helper itself, so it would still pass if the trainer moved interruption before publication. Add a trainer-level or narrowly factored test that fails if interruption precedes the complete paired publication. The current source does call interruption after publication at `train_adapter.py:258-259`; this is a regression-test weakness, not an observed ordering bug.
- **Skip-it:** gradient checks correctly operate after scaler unscale, reject nonfinite values and a zero aggregate norm, and require an update at each step. Default FP16 scaling may fail a real smoke on overflow; failing closed here is consistent with qualification, not a reason to silently claim a successful update.
- **Skip-it:** the resumed path constructs a new base model, loads the saved adapter, validates optimizer identity, restores optimizer/scaler/CPU/CUDA RNG, and executes remaining step 2. The completed-finalization shortcut cannot satisfy the orchestrator's required step-1 partial snapshot.

## Parsimony

No must-fix. The fixed smoke configuration, small interruption hook, reused immutable checkpoint publication, and nested comparison serve this bounded qualification without speculative infrastructure. No new dependency or unrelated product feature was introduced.

## Product

The implementation covers actual forward/backward instrumentation, parameter residency, token and gradient/update evidence, atomic paired saves, partial reload, optimizer/RNG comparison, and explicit ephemeral artifact status. Model/runtime qualification remains pending the parent-run actual GPU experiment. CPU control tests and imported dependency verification alone provide no GPU, quality, or Arabic-remediation claim. Metrics comparison is the narrow missing requirement identified above.

Validation: independently ran `tests.test_lora_qualification` with bundled Python 3.12.14: 6 tests passed. The supplied broader 95-test gate was not rerun by this judge. Tensor numerical branches require actual Torch/runtime execution and were not claimed exercised on this CPU environment.

panel: full — correctness ✓ fresh-context Codex combined · parsimony ✓ fresh-context Codex combined · product ✓ fresh-context Codex combined (browser not applicable) · security — separate judge · d1 — not used for required gate · GPU qualification — pending

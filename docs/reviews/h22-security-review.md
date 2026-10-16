# H22 independent security boundary review

- Baseline: `11a3a5359afa896787198e0492ccd036d0b6f4f8`; uncommitted Hala-precision source/config/freeze/test delta.
- Security verdict: PASS; concrete security findings: [].
- Coverage: explicit provider admission, exact ephemeral cache and cache-environment checks, externally pinned source/config freeze, clean source identity, complete external adapter verification before ML allocation; concrete precision profiles, imported package pins, disabled remote model code, CUDA-only device map/residency and absence of external factory CLI.
- Preservation: focused test verifies complete SETTINGS/QwenInfer/training-guard AST and unchanged H19 development source bytes against baseline. Historical precision helper remains unchanged. Default FP16 branch retains its model loading arguments.
- Verification: 35 focused precision/provider/adapter/runtime CPU tests passed in 2.189 seconds. Includes local rejection, wrong external freeze, cache escape rejection, malformed adapter identity, exact loader arguments, frozen profile strictness, pairing identity and sampler failure preserving inference evidence.
- Operational observation for combined review: `disk_bytes` recursively walks the entire cache without an entry/time budget, after runtime finalization cancels the outer alarm. Exceptions become null/error telemetry, but traversal duration itself is unbounded. This is a bounded-cost acceptance concern; no independent attacker boundary was established.
- Limits: CPU source review does not qualify actual package/quantized-adapter compatibility, GPU residency or representative quality. No GPU use, ML installation, weights, external communication, billing or source edits performed.

## Repair delta verification

- Actual verified prospective freeze SHA256: `a8ed7f28ddbf164bee345acbc95c5b2a374b832e92cfa94562043b2e6a943e80`.
- 38 scoped CPU tests passed in 2.373 seconds. Source/config hashes validate against that digest.
- Finalization observation resolved for ordinary local filesystem traversal: incremental scandir limits measurement to 100000 entries and checks five elapsed seconds between filesystem operations. Exhaustion produces null cache bytes with TimeoutError while preserving inference evidence. These are cooperative limits, not hard preemption of an individual blocking filesystem call.
- GPU query now targets the UUID of actual logical cuda:0; unavailable identity disables device sampling with a label. Unsupported fields retain null plus individual labels; cadence metadata describes minimum wait rather than a fixed sample interval. No new security findings.
- Remaining boundedness observation: admission `check_cache` still uses unrestricted `approved.rglob('*')`, twice before the outer alarm is installed. This symlink validation walk has no entry/time budget. The finalization repair does not bound that earlier walk; communicated to the parent combined review.

## Final admission repair

- Remaining admission observation resolved: `check_cache` now invokes the same incremental scandir validation with 100000-entry/five-second cooperative limits. Exhaustion propagates before dependency metadata lookup or ML execution; it does not approve a partially scanned cache.
- Every encountered entry is resolved and checked inside the approved root before type/stat handling. External directory symlinks reject; directory symlinks are not traversed. Root cache symlinks remain rejected. Individual blocking filesystem calls remain outside cooperative preemption, as already noted.
- Focused admission-budget, measurement-budget, cache-rejection and unique-file tests: four passed in 0.128 seconds. A native Windows symlink probe could not run because creating symlinks returned WinError 1314; no claim of native symlink execution coverage.
- Actual source freeze verified: `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b`; helper SHA256: `1fb95afef7254b73f7229ec0da5e6ad6d2e91b42f42e5b4e0476f56b75acd9f5`.
- Final delta security verdict: PASS; findings: []. No GPU, ML downloads or source edits.

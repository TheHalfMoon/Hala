# H22 independent combined review

Scope: uncommitted H22 source in `work/Hala-precision`, branch `ps/h22-development-precision`, against `11a3a5359afa896787198e0492ccd036d0b6f4f8`. Correctness, parsimony and product reviewed independently of builder assertions. Source review only; no GPU, model download or source edits. Actual H21 runs do not qualify H22.

## Must fix

- `evaluation/development_precision.py:100`: the observer always queries physical GPU index 0, while the model and allocator measurements use logical `cuda:0`. With `CUDA_VISIBLE_DEVICES=1`, a GPU UUID mapping, or reordered devices, valid-looking used/free/utilization/power readings can describe another GPU. Resolve and record the physical UUID of the actual CUDA device, use it for the external query, and fail open with labelled null readings if identity cannot be established. Add a remapping test. This blocks truthful matched device-cost evidence, rather than model inference.

## Worth considering

- `evaluation/development_precision.py:104`: unsupported individual fields become null without an availability/error label. A focused mock returned `power_watts=None` and no `device_error`; add per-field unsupported labels to meet the spec's missing-telemetry reporting rule.
- `evaluation/development_precision.py:138`: `interval_seconds=2` is a wait between completed samples; each query may add up to two seconds. Rename it to an explicit wait/poll delay or report the actual timestamp spacing so consumers do not assume a fixed two-second cadence. Timestamps already permit accurate analysis.

## Confirmed clean / skip it

- Runtime seam reuses the same QwenInfer, SETTINGS, runner, parser, simulator and verifier. Existing test compares callback/settings/remote-guard AST and all H19 development source against the immutable base. Historical H14/H17 files have no source diff.
- Profiles pin actual int8/NF4 configurations, reject absent quantized module classes, prohibit CPU offload, and reuse the full-model/vision/quant-state CUDA-residency check. Actual pinned-package/model/PEFT compatibility remains a GPU qualification gate.
- Admission, external source/config digest, settings/bundle, clean commit, cache and optional exact archived-adapter checks precede model allocation. No task object crosses the inference callback.
- Pairing binds source/model/package/adapter/task order, rejects duplicate/reordered/missing reported tasks and inconsistent summaries, preserves English/Arabic semantic/final-state groups, and makes no adoption claim. Adaptive prompts need not be identical after divergent responses; their shared frozen construction is the appropriate invariant.
- Sampler errors preserve inference evidence, subprocess timeout and sample count are bounded, storage distinguishes underlying tensor/cache bytes from allocator/device observations, and no new general abstraction is warranted.

## Independent validation

- `python -m unittest tests.test_development_precision tests.test_development_runtime tests.test_development_adapter -q`: 30 tests passed in 2.567 seconds.
- `python -m unittest tests.test_quantization -q`: 8 tests passed in 0.073 seconds, covering reused historical helpers without editing them.
- Focused CPU-only probe with `CUDA_VISIBLE_DEVICES=1` and a mocked external response confirmed the command still contains `--id=0`; unsupported power became null with no label. No real GPU query or ML import was used by this probe.
- Prospective freeze verifies on the current bytes through the focused suite. Git `core.autocrlf=false`; CRLF configuration bytes are currently preserved. Signed source, external operator freeze, full repository/Jev/Graft/security gates and actual H22 runs remain the conductor/operator's responsibility.

Panel contribution: combined correctness/parsimony/product, session model; separate security review belongs to the parent panel. Verdict: fix device attribution before H22 execution or claiming complete cost evidence.

## Delta re-review — repairs resolved

Reviewed the repair delta against the findings above. Current Git HEAD remains `11a3a5359afa896787198e0492ccd036d0b6f4f8`; H22 changes are uncommitted. Exact reviewed content identities:

- Prospective freeze SHA256: `a8ed7f28ddbf164bee345acbc95c5b2a374b832e92cfa94562043b2e6a943e80`.
- Precision module SHA256: `e0574ea0e2c84f9d1ec9a14c17df613ec01e2f948f9da940d051a5c2e7e96327`.
- Shared runtime SHA256: `20b3bf32ca37107ed05e9f2c3ed74413653b51107f5abf474f4a40b25ef58895`.

Resolved: logical `cuda:0` properties provide the physical GPU UUID used by `nvidia-smi`; unavailable/invalid identity leaves labelled null device telemetry instead of guessing an index. The runtime records the UUID and passes it to the observer before model loading. Unsupported individual fields carry explicit labels. Sampler metadata accurately describes the minimum wait instead of promising a fixed cadence.

The additional cache scan now has entry/time budgets with streaming directory iteration. Its timeout is caught by the existing fail-open finalization path. The repaired runtime regression explicitly forces that timeout for int8 and confirms both calls, measured status and independent summary survive.

Independent delta validation: `python -m unittest tests.test_development_precision -q` passed all 15 tests in 1.215 seconds. The tests include remapped-device UUID querying, no query without identity, per-field unsupported labels, wait metadata, cache entry/time budgets, and persisted inference evidence under telemetry failure. Independently recomputed and verified the freeze and content hashes above. No GPU or local weights used.

Remaining findings: none in this delta. The combined-review source blocker is resolved. Actual GPU compatibility/quality/cost qualification and the other mandatory source gates remain separate; this review does not qualify model precision or adoption.

## Final delta re-review — admission and paired physical identity

Reviewed only the final cache-admission and paired-identity repairs. Verified prospective freeze SHA256 `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b` and precision helper SHA256 `1fb95afef7254b73f7229ec0da5e6ad6d2e91b42f42e5b4e0476f56b75acd9f5`; runtime hash remains the previously reviewed `20b3bf32ca37107ed05e9f2c3ed74413653b51107f5abf474f4a40b25ef58895`.

Cache admission now invokes the bounded incremental scan and propagates exhaustion before dependency lookup or `_run`; it never accepts an incompletely inspected cache. The documented time bound is correctly qualified as checks between OS operations, which cannot preempt a stalled filesystem syscall. Paired identity binds the actual GPU UUID, rejects differing physical devices, and explicitly marks the physical-device comparison unqualified when both identities are unavailable. These changes satisfy the requested source behavior without adding an alternative execution path.

Independent targeted validation: the cache-admission exhaustion and paired-identity tests both passed (`Ran 2 tests in 0.070s`). Inspected their assertions for entry/time exhaustion, dependency/runtime nonexecution, differing resolved devices, and both unknown identities. Verified all freeze entries through `verify_sources`. No broad retest, source edit, GPU execution or model download.

Remaining findings: none in the final delta. Combined correctness/parsimony/product source review remains clear; actual precision qualification is still pending.

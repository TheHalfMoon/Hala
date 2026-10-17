# H21b independent combined review

**Changes required: one P2 failure-evidence must-fix before the actual baseline.** Independent fresh combined correctness/product/parsimony review of `evaluation/development_runtime.py`, its tests, H21b docs/spec and current H19 callable interfaces. Separate security PASS was read and does not substitute for this correctness bar. No tracked source edits, full-suite rerun, ML import/install, weight download or GPU execution occurred.

## Must-fix

- **`QwenInfer.__call__` finalizer and `run` finalizer: telemetry failure can discard actual call evidence and prevent the blocked-result checkpoint.** The per-call finalizer assigns `record['memory_after'] = self.sample(...)` before appending the record. A legitimate CUDA error can also make `mem_get_info`/allocator queries fail; a fake sampler returning the initial sample then raising RuntimeError reproduces **model_called=true, sink_records=0**, even though a raw model completion already existed. This also masks an original model exception if both generation and post-call sampling fail. The outer `run` finalizer similarly calls `memory_sample` before `persist`; if that sampling fails, caught-failure result status/calls may never be saved and cleanup is skipped. Make memory telemetry best-effort with explicit unavailable/error metadata, preserve an original call exception, and ensure completed/failed call records append and result checkpointing occur regardless of telemetry availability. Do not invent zero memory numbers. Add a regression for post-call sampler failure and one showing a blocked result still persists when final CUDA telemetry fails. Keep the overall experiment blocked when generation actually fails.

Reproduction: `work/h21b_combined_probes.py`, run from task workspace using `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21b_combined_probes.py`. The reproduction changes only the injected fake sampler and writes no repository data.

## Assessed bars

| Bar | Result |
|---|---|
| Correctness/integration | Latest H19 requested-kind and semantic metrics are aggregated separately from final completion. Callback uses public text/image inputs, proper input-length trimming and untouched raw decode. Eight focused tests independently passed in 0.192 seconds. The failure-evidence issue above prevents unconditional pass. |
| Reproducibility/settings | Exact reviewed dictionary binds immutable base revision, FP16/SDPA, greedy seed42, deterministic flags, max128 output/4096 input tokens, 20s per generation and 16 model calls/eight actions. Altered/missing/extra settings and Boolean substitutions reject. External media manifest SHA, source implementation hashes, clean expected Git commit, pinned font/package metadata and remote guard precede ML imports/allocation; imported versions rechecked. Actual GPU/determinism compatibility remains unexecuted. |
| Source/publication | Source must be published/clean before real execution, as intended. H19 frozen implementation and settings are externally bound. Git cleanliness is deliberately scoped to H19/runtime source, not an attestation of arbitrary environment/package provenance. Source checks are operational consistency, not hostile-host protection. |
| Counter semantics | Independently exercised mixed-language synthetic results: two episodes/one completion, five Mind calls/one semantic correct, all-language Mind accuracy0.2, separate standalone Mind validity/exact correctness and recovery/environment counters. `model_calls` in language summaries counts episode calls; standalone Mind cases are reported separately and global raw call log includes them. Do not present that episode-only count as all-run total model calls. |
| Raw outputs/latency | Raw completion is not repaired or parsed inside GPU callback. Global sink records prompt/image digest/output tokens and preprocessing/generation/wall seconds; H19 records final transitions/state. Injected public text-only calls confirmed no image argument, sequential call indexes and cold-first-call flag. Telemetry loss under error is the concrete repair above. |
| Containment | Remote opt-in/actual Colab marker and explicit Kaggle rejection protect accidental execution; resolved approved caches and current-frame directory checks are scoped operator boundaries. Adapter input rejects before load. Monotonic deadlines stop later calls after a caught Python alarm; external worker timeout remains needed for hung CUDA, correctly documented. |
| Product/data | Supports an unchanged-base bilingual Mind and synthetic closed-loop baseline, without native execution or authority. All ten standalone Mind cases are authored test cases even for `--split dev`, clearly labelled. They must not become prompt/training/selection feedback while still claimed independent. Any later shared adapter comparison must retain exact held-out tasks and settings or introduce a newly frozen protocol. No real quality/bilingual/media result is claimed by these fakes. |
| Parsimony | The small optional remote wrapper, explicit pins and existing H19 callbacks are proportionate. Repair failure-path logging locally; no need for a generic service framework, retry mechanism or altered parser. |
| Memory/cost evidence | Allocator peaks and sampled whole-device usage/RSS are distinguished accurately. Initial/load/final samples and before/after call samples are useful; samples are not continuous device peaks or full serving cost. Memory availability should be explicitly represented rather than allowed to erase primary evidence. |

## Verification and limits

Read source/tests/spec/docs and H19 freeze/runner integration. Independently ran the eight runtime tests and separate fake-only probes for sampler failure, call ordering/raw public inputs and mixed-language summaries. No full-suite duplication or actual model experiment. Existing security review covers its own boundaries; Jev/Graft/CI/signing and the reviewed remote media freeze remain root gates. After repair, run a separate worker on the published source with an externally archived manifest and visibly verify Arabic frames before actual baseline inference.

Reviewed source hashes (match the earlier security review):

```text
evaluation/development_runtime.py a68a959e5d0a213ef6d2f6bb731b1437f760486abf617d0be4ce5be9b75f3ad2
tests/test_development_runtime.py 9dd444dcf739939b3820dafb80454460b6c2ece89917574702bae4f946e6b71e
```

Panel: fresh independent combined correctness/parsimony/product; one must-fix with executed reproduction; no source edits or release authorization.

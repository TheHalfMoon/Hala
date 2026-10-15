# H21b independent combined telemetry delta

**PASS for the repair delta.** The original evidence-loss must-fix is resolved. No new must-fix found in these changes. Original combined review's scope, measurement limits and real-runtime prerequisites remain applicable.

Fresh independent reviewer read `record_memory`, `finalize_result`, call/load/final integration and new regressions. No tracked source edits, full-suite rerun, ML imports, model downloads or GPU experiments.

## Executed independent replay

- Replayed the original sampler-failure probe using the actual callback and fake model seam. A successful model call followed by failed post-call telemetry now **returns the untouched raw completion, appends one sink record, stores memory_after=null and records RuntimeError as telemetry availability metadata**. The former result was zero records and a replaced completion exception.
- Injected failures into both before/after sampling while generation raised its distinct `RuntimeError('fake device failure')`. The original generation exception/message propagated; its call record was appended with error_type=RuntimeError and both memory samples null. Sampling ValueErrors did not replace the primary error.
- Passed that blocked result to the actual finalizer while final sampling raised RuntimeError and cache cleanup raised OSError. **A valid blocked-result checkpoint still persisted with its original error and call**, memory_final=null, both telemetry/cleanup error types, and max_sampled_device_used_bytes=null because no valid sample existed. No invented zero memory measurement appeared.
- Existing valid sample aggregation remains covered by the targeted test (77 bytes survives unavailable final telemetry). Independent mixed-language summaries and public text/cold-call probes continued to pass.
- Independently ran all **10 focused runtime tests: passed in 0.244 seconds**. No full-suite repetition.

Reproducible outside-checkout probe: `work/h21b_combined_probes.py`, run from task workspace using `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21b_combined_probes.py`. The original sampler setup is retained, with assertions updated to require the repaired behavior and an independent blocked-checkpoint probe added.

## Review bars

- **Correctness:** before/after/load/final telemetry is best-effort with explicit missing values. Completed/failed inference evidence and the original model exception remain primary. Finalizer tolerates unavailable device samples and cache cleanup without suppressing persistence.
- **Product/evidence:** device telemetry failure no longer implies a fabricated memory number or erased call. A generation error still causes blocked/error evidence, not a falsely qualified successful experiment. Primary model and semantic outcomes retain their prior definitions.
- **Parsimony:** the small sampling wrapper and extracted finalizer localize the repair; no retry, parser change, authority grant or broad infrastructure was added.
- **Limits:** actual CUDA behavior, force-killed/hung kernels, disk failure, VM loss and Arabic/model quality remain unqualified. Atomic persistence can still report genuine filesystem failures; these tests establish caught telemetry/cleanup failure handling, not universal crash recovery. Separate security delta, Jev/Graft/CI/publication and a reviewed actual remote run remain conductor gates.

Exact reviewed SHA256:

```text
evaluation/development_runtime.py 7c8f0edfede7bf40a375fc3fcc74043c852700dc0fe405566a93b04c8eafa616
tests/test_development_runtime.py f1e68692123ffa910c4d9f96a76f2264e8f719a1418474a5d5aff1c1af40f90a
```

Panel: fresh independent combined correctness/product/parsimony delta; original failure reproduced against the repair, no source edits or release authorization.

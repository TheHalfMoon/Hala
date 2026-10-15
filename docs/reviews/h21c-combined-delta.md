# H21c independent combined repair delta

**PASS for the reviewed delta.** The original telemetry/evidence-loss must-fix is resolved. Baseline hashing/parsing and Type image reads are now explicitly bounded. No new must-fix found in these repairs; original review's real-runtime, numerical, data-coverage and persistence limits remain.

Fresh independent reviewer read current `record_memory`, `record_completed_step`, `finalize_training_result`, trainer integration, baseline read/preflight, bounded Type media read, regressions and docs. No tracked source edits, full-suite rerun, ML installation/import, weights or GPU execution.

## Independent executed probes

- Adapted the original actual-finalizer AST replay to invoke the real new helper while its telemetry sampler raises RuntimeError. Raised an original training ValueError in the enclosing try/finally. **The exact original exception object propagated**, while the actual `result.json` was atomically saved with blocked/original error, memory_final=null, max_sampled_device_used_bytes=null and bounded exception-class metadata. The private sampler message was absent.
- Called the actual completed-step helper with a measured loss/norm/112 changed-tensor metric and failing sampler. The persisted progress retained that numerical step with memory=null and telemetry-error class. Source confirms all real successful optimizer steps call this helper, and load/final sampling also uses best-effort recording.
- Verified JSON hash and parse use the same bounded bytes: correct external hash parsed; changed bytes rejected; a deliberately invalid-JSON file with mismatched hash was rejected **before reaching the parser**, independently instrumented with a failing parser sentinel. The 16MiB upfront stat/read bound and source mutation-length checks are present; oversized preflight rejection is covered by the focused regression.
- Replayed positive Arabic Type encoding: only public prompt+image passed to the processor, both full/prefix calls use the same image, completion prefix is masked. Type read now checks size before opening and caps the read at2MiB+1, then validates exact length/signature/digest.
- Replayed held-out Mind diagnostic independence: making all standalone test diagnostics correct still leaves the development deficit gate unchanged. This does not prove actual baseline provenance; the externally archived real-run record/operator gate remains required.
- Independently ran **10 focused training contract tests: passed in6.909 seconds**. No full-suite rerun or fake claim of executing Torch updates.

Reproduction and assertions: `work/h21c_combined_probes.py`, run from task workspace with `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21c_combined_probes.py`. The original finalizer AST probe is retained and updated for the repaired helper, with original-exception identity, persisted fields and completed-step checks.

## Bars and remaining gates

- **Correctness/product:** telemetry availability no longer decides whether measured numerical progress or original failure evidence exists. Missing samples are explicit null/error classes, not fabricated memory values. Completed optimizer metrics persist before proceeding; failed training remains blocked.
- **Integrity:** externally expected baseline digest is checked against exactly the bytes then parsed, before bundle projection; bounded reads prevent large input hashing/loading before rejection. PNG identity/media and train-only bilingual projection retain their earlier checks.
- **Parsimony:** local helpers and bounded readers are proportionate to this fixed116-step pilot. No retry/schedule mutation, broad importer, native action, learned authority or credential transport was added.
- **Limits:** actual forward/backward, PEFT save-format compatibility, mask/image-token behavior, durable signed artifact retrieval and fresh-session load/evaluation remain unexecuted. Finite guards are source inspected/qualified-helper reuse, not a new GPU result. Filesystem persistence failure or process/VM loss can still interrupt evidence; caught telemetry errors are the behavior exercised here. Latest reviewed H21b implementation must be integrated before freezing source/data/runtime identities. Separate security delta, genuine Jev/Graft, exact-source CI and signed source/artifact publication remain conductor gates.

Exact reviewed SHA256:

```text
training/development_data.py cc0fa09eedab538ba1ef3b412212fc16229ac9ecf93c97cdc3ff1b30a1b1845a
training/scripts/train_development.py 17f0270cb721d343aecb5dcab1d35b3c88fcdf3735e3cccf2e180189bd7790f1
tests/test_development_training.py 91633217b1e8ed0a9441fc02b0a883fa00c23db54ed214b3b94f0f94ea1d4e68
```

Panel: fresh independent combined correctness/product/parsimony delta; original failure replayed against repaired implementation, no source edits or release authorization.

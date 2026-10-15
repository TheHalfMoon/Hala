# H21d independent combined review

**PASS for the scoped implementation; no must-fix found.** This is source/CPU compatibility evidence, not actual adapter loading, persistence, quality or release qualification. Fresh independent correctness/product/parsimony judge reviewed AGENTS/CLAUDE, H21d spec/docs, the new adapter module, runtime diff against `c7ed4050e00f9c358783ce76440b753110649470`, archive contracts and new tests. No tracked source edits, ML installations/imports, GPU tasks or model downloads.

## Worth considering

- **Precision reporting:** official PEFT0.20 defaults `autocast_adapter_dtype=True` and promotes float16/bfloat16 adapter weights to float32. The source currently relies on that default. This is compatible with the qualified FP32 LoRA producer and leaves the pinned base FP16, but it is not evidence that every candidate tensor executes/stores in FP16. Record the actual loaded adapter dtype counts and describe the candidate as FP16 base plus observed adapter precision before full-cost/precision conclusions. Do not silently downcast the trained adapter merely to match an imprecise label. Current loaded-value comparison correctly compares against archive tensors converted to the actual loaded tensor dtype; that is value equivalence at loaded precision, not a claim that loaded tensor bytes equal archive bytes.

## Confirmed clean / skip-it

- **PEFT API compatibility:** independently inspected the exact official0.20 wheel in memory, SHA256 `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233`. `PeftModel.from_pretrained` accepts an existing model and PathLike adapter directory, `is_trainable=False`, and extra keyword arguments; `local_files_only` is supported by the load path. `get_peft_model_state_dict` is exported at package root, defaults to adapter `default`, and supports `save_embedding_layers=False`. No speculative API shim or config rewrite is needed. The inspection does not execute PEFT or establish actual next-checkpoint compatibility.
- **Identity/load boundaries:** missing/partial external adapter flags reject before ML imports; exact archive manifest and all five external identity fields are verified. Only trained-adapter kind enters inference. Bounded original bytes are copied to a private temporary snapshot and reverified; caller changes to the original afterward cannot alter the loaded snapshot. Config bytes, including null revision, remain untouched. Base loader still receives the exact immutable base revision explicitly.
- **Inventory/values:** actual safe-format decode is on CPU after remote/package gates and before base allocation; all112 expected shapes, supported float dtypes and finite values are checked. PEFT loading is followed by actual state-dict inventory/shape/finiteness/value checks and requires every parameter frozen. This rejects warning-only missing weights, extra initialized modules and changed values; load errors propagate without baseline fallback. No additional inventory abstraction is needed for the pinned base.
- **Baseline comparability:** independently compared Python AST against the exact base commit: **SETTINGS and the complete QwenInfer callback are unchanged**. H19 prompts/data/renderer/verifier were not modified. Same public-input generation path, same selected cases, same scoring and raw-output preservation are used for baseline/candidate. Runtime source identity changes prospectively, before new baseline outputs; the later training prerequisite intentionally requires that exact runtime hash. Do not attempt to relabel an old-runtime baseline as this new source.
- **Honest classification/metadata:** unchanged-base path remains original schema, adapter_loaded=false and training_runs=0. Candidate starts with distinct adapter-evaluation schema and external manifest/identity/file hashes, sets adapter_loaded=true only after successful attachment, retains training_runs=0 and false quality/native/authority flags. Failed attachment remains blocked and cannot masquerade as unchanged-base evaluation. The producer commit is distinct from the evaluation commit, correctly documented.
- **Parsimony/product:** a small optional adapter seam and minimal shared-runtime integration are proportionate. No arbitrary URL download, credential path, pickle intake, optimizer restore, native executor, learned authority or separate model architecture was introduced. Operator-owned immutable artifact retrieval and externally archived trust anchors remain separate real-world gates.

## Executed checks

- Independently ran **all eight new adapter tests: passed in1.234 seconds**. These meaningfully cover no-import invalid identities, exact private snapshot, harmless-probe rejection, all-tensor finite/inventory defects, safe-load/PEFT order and flags, changed weights/trainable rejection, error propagation, unchanged baseline classification and complete fake-runtime case/load-order/result metadata. They remain fake CPU seams; no real model behavior is inferred.
- Official-wheel/API and baseline AST checks are reproducible in `work/h21d_combined_probes.py`; complete small results in `work/h21d-combined-probes.json`. Run from task workspace using `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21d_combined_probes.py`. One initial probe assertion assumed an unannotated signature string; it was corrected to inspect the AST default value directly. No implementation defect was inferred from that probe formatting mismatch.
- No full-suite duplication; builder's206-test result remains builder evidence. Separate security review, genuine Jev/Graft, exact-source CI and signed publication remain conductor gates. Real remote safetensors/PEFT load and identical frozen bilingual evaluation must still be performed on the independently retrieved retained artifact; this review does not prove its existence or quality.

Exact reviewed SHA256:

```text
evaluation/development_adapter.py b78d84af1073369a3f8ef947b2bfce56e0fc41b2580f2dec5d0e54123132a248
evaluation/development_runtime.py da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35
tests/test_development_adapter.py 2b49e5fdcdf61f056808f637c4c2cae834bfc72a3dcc97f01c3db21eba10b040
```

Panel: fresh independent combined correctness/parsimony/product; no source edits, no merge/deployment/release authorization.

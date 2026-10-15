# H21d independent security/integrity review

Verdict: PASS; no actionable security or data-integrity findings in the scoped change against `c7ed4050e00f9c358783ce76440b753110649470`. This judge did not author H21d or change repository source. No ML installation/import, GPU execution, large-weight download or full-suite rerun occurred.

Reviewed AGENTS.md, CLAUDE.md, H21d spec/docs, new `evaluation/development_adapter.py`, runtime diff and targeted tests. Existing archive validation was reviewed as context, not expanded into a new archive audit.

## Concrete boundary evidence

- Every adapter argument and full externally pinned identity must be supplied together. The archive verifier rejects mismatched immutable identities/digest and non-trained probe kind before ML loading. Caller trust remains the independently obtained signed archival record; the verifier does not infer its own trusted anchors from downloaded bytes.
- Snapshot copying uses only already allowlisted filenames, caps every read at its fixed file limit+1, preserves original config bytes and revalidates the private copy against the external manifest digest. Initial source symlinks/reparse points are rejected by the existing archive guard. An independent probe mutated the original config immediately after its first successful archive verification but before copying: the copied snapshot failed revalidation and was never yielded. Original-source changes after a valid snapshot cannot change its private bytes.
- Actual safe-format CPU decoding occurs after package/remote gates and before base allocation. Complete 112-tensor inventory, exact shapes, allowed floating dtypes and finite values are required. No pickle/config rewriting/arbitrary URL loading is introduced.
- PEFT receives the private snapshot, `is_trainable=False`, `local_files_only=True`, then `eval()`. The snapshot is reverified immediately before loading. Complete actual loaded adapter state is validated and compared to the independently decoded expected tensors in loaded dtype. Missing keys, extra vision modules, nonfinite values, changed/random values and trainable model parameters reject without a baseline fallback.
- Runtime clean-source checks include the adapter helper and archive implementation. Metadata and actual imported PEFT0.20.0/safetensors0.6.2 versions are pinned in candidate mode. Immutable base revision, remote-code rejection, SETTINGS, prompts, cases and generation path remain unchanged.
- Candidate results use a separate schema and record exact archive identity, manifest/file digests and lengths. `adapter_loaded` becomes true only after validated attachment; blocked attachment retains false. `training_runs=0` truthfully describes this evaluation process. Native/quality/execution flags remain false.

## Independent probes

`work/h21d_security_probes.py` ran through bundled Python and `rtk proxy`, exit 0. It independently exercised source-mutation timing, silently randomized loaded values, additional vision adapter, missing loaded tensor, nonfinite loaded tensor and trainable parameter rejection. It also replayed the targeted fake tests for incomplete/mismatched external identity before ML imports, harmless-probe rejection and full baseline/candidate load order, identical task selection and truthful saved result classification. Fake tensors/modules provide boundary evidence only, not actual PEFT/GPU behavior or quality qualification.

Reviewed SHA256:

- `evaluation/development_adapter.py`: `b78d84af1073369a3f8ef947b2bfce56e0fc41b2580f2dec5d0e54123132a248`
- `evaluation/development_runtime.py`: `da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35`
- `tests/test_development_adapter.py`: `2b49e5fdcdf61f056808f637c4c2cae834bfc72a3dcc97f01c3db21eba10b040`

Real fresh-session retrieval, actual loaded tensor/value checks on the remote pinned stack and unchanged-task quality comparison remain unexecuted operator/model gates. This review supplies independent PStack security evidence only.

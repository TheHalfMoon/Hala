# H21a independent combined repair delta

**PASS for the reviewed offline archive repairs; prior producer-config incompatibility is resolved.** No new must-fix found in this delta. This verifies bounded source/config/format compatibility, not a real saved-adapter upload, actual PEFT serialization/loading, GPU execution or model qualification.

Fresh independent combined judge read the original `h21a-combined-review.md`, current archive/tests, qualified training producer, pinned Qwen config evidence, spec and docs. No tracked source edits, commits, ML imports, installations or model-weight downloads.

## Independently executed checks

- All **13 targeted archive tests passed in 4.793 seconds, one skipped**. The real Windows symlink-creation test skipped for unavailable account privilege. The real Windows junction test executed successfully while temporarily removing `Path.is_junction`, confirming `lstat().st_file_attributes` detection does not depend on that newer pathlib method. Full suite was not repeated.
- Downloaded only the official **775,777-byte PEFT0.20.0 wheel into memory**, independently verified SHA256 `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233`, and inspected config source AST without installing/importing the package. All **41 official inherited/dataclass fields** are allowlisted. The saved producer fixture contains all those fields except `runtime_config`, which official save serialization removes. The exact qualified language-only target regex now passes unchanged.
- Verified the five previously missing serialized fields against official source: `peft_version` is auto-filled from its null default; `lora_ga_config`, `monteclora_config`, `use_bdlora` and `velora_config` default to null. The archive requires version0.20.0 and keeps those four variants inert. A full official-default/qualified-override fixture archived and verified successfully, preserving its original config bytes exactly.
- Independently rejected **ten mutations**: broader target regex, version0.21, nonnull four variant configs (including `use_bdlora=false`, deliberately distinct from its required null default), integer inference mode, enabled DoRA, target parameters, and PiSSA initialization.
- Checked pinned official Qwen text dimensions `(layers, hidden, query_heads, value_heads, head_dim) = (28,2048,16,8,128)`. Current exact inventory is **112 tensors /802,816 elements** with A `[4,2048]`, q B `[2048,4]`, v B `[1024,4]`. Targeted tests independently derive that inventory from checked-in official config bytes and confirm the producer regex. Partial inventory, layer99, alternate prefixes and wrong shapes are rejected; contiguous complete payload and A/B dtype-pair checks remain intact.

Reproducible independent probe: `work/h21a_delta_probe.py`; results: `work/h21a-delta-probe.json`. Run from task workspace with `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/h21a_delta_probe.py`. Prior review's failure artifacts were preserved rather than overwritten.

## Review bars and compatibility limits

- **Correctness/product:** accepting the exact producer regex and complete official inert defaults fixes the concrete P1 rejection without rewriting producer files or widening the regex. Complete pinned tensor inventory closes the partial-adapter format gap. Reparse detection implements the intended Windows/Python3.11-compatible directory boundary.
- **Parsimony:** explicit reviewed fields/shapes and one portable reparse helper fit this narrowly pinned base/adapter. No generic regex admission, arbitrary variants, additional executable fields, ML deserializer or network path was added to the archive module.
- **Evidence:** fixture acceptance plus official source inspection proves the reviewed offline compatibility seam. It does **not** prove the actual next GPU producer emits those exact names/config bytes or that PEFT can load a downloaded bundle. Actual remote save/archive/download/load/evaluate must run before declaring persistence qualified. Do not silently remove unsupported producer fields or rename tensors to fit this checker.
- **Limits accurately documented:** no tensor finiteness/training-quality proof; no pickle/optimizer/base-weight acceptance; caller-owned exclusive directories, no defense against concurrent filesystem races; external signed-commit identities/manifest hash still required. Missing Windows symlink privilege is disclosed rather than represented as a successful real-symlink test. Separate security review, genuine Jev/Graft, CI and publication remain conductor gates.

Exact reviewed SHA256:

```text
training/adapter_archive.py 95773e65224b6ee57c89f7edd61a3d7b247cd623eef3cca4637eed7cbc6abc55
tests/test_adapter_archive.py 809caf2d551eb7e0991bb5935efee73b4ee88376adff26290c4bdfc4a5b48227
```

Panel: fresh independent combined correctness/product/parsimony delta; no source edits, no release or merge authorization.

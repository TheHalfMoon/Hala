# H21a independent security/integrity PStack review

Verdict: address one P2 fail-open portability finding before publishing this verifier as supporting Windows Python 3.11+. No arbitrary execution, credential or network path found in scoped code.

Reviewed `training/adapter_archive.py`, `tools/archive_adapter.py`, `tests/test_adapter_archive.py`, `specs/H21a-adapter-archive.md`, `docs/H21a-adapter-archive.md`, AGENTS.md, CLAUDE.md and PRODUCT.md. Source unchanged. No model loading, downloads, GPU operations or full-suite repetition. The expected manifest hash and identity are trusted external inputs as required; no finding assumes replacing those anchors.

## P2: Windows Python 3.11 silently disables junction rejection

Locations: `training/adapter_archive.py`, `_root` and `_read_directory` optional `is_junction` checks.

The documented runtime is Python 3.11+, but `Path.is_junction()` was introduced in Python 3.12. Both checks use `getattr(..., lambda: False)`, so the unsupported detection API becomes a clean bill of health. A Windows junction is not a symbolic link according to `is_symlink()`; `is_dir()`, `iterdir()` and file reads follow it. A junction source directory or parent therefore passes the archive boundary on Python 3.11, despite the spec requiring junctions to fail closed. Destination parents and downloaded-bundle roots use the same guard.

Concrete probe: created a real Windows directory junction with PowerShell `New-Item -ItemType Junction`. Current bundled Python 3.12 correctly rejected it. Temporarily removing `Path.is_junction` from the in-process API to reproduce Python 3.11 availability caused `create_bundle(real_junction, ...)` and subsequent verification to succeed. This was an actual filesystem junction; only runtime API availability was simulated. The probe restored the API and removed only the junction itself afterwards. No real Python 3.11 interpreter was available/tested.

Minimal fix: on Windows inspect `lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT` for each component/file, conservatively rejecting reparse points; alternatively explicitly require Python 3.12+ on Windows and fail immediately when junction detection is unavailable. Add the missing-API regression rather than substituting `False`.

## Boundary evidence and limits

- Under the current Python 3.12 API, the real junction source is rejected. Existing strict flat filename allowlisting rejects explicit path/nesting additions; lengths and total bytes bound reads. Documented exclusive directory ownership excludes concurrent filesystem mutation from this tool's contract.
- An adapter byte substitution under an unchanged trusted manifest digest was rejected.
- Oversized safetensors header, trailing payload and duplicate header keys were rejected in independent bounded probes. Header offsets are range checked and then required to form contiguous nonoverlapping payload coverage; unsafe deserialization/imports are absent.
- Config field allowlisting and inert JSON parsing expose no execution path in this phase. Optional metadata is producer evidence only. `quality_qualified` must remain false; manifest-only pretraining archives cannot claim an adapter.
- A separate scope probe accepted a q-only layer99 A/B pair with `[4,1]` and `[3,4]` shapes. This confirms validation checks bounded rank and A/B presence, rather than Qwen layer/dimension completeness. Documentation explicitly disclaims model compatibility, so this review does not count that as a separate security finding. Subsequent remote loading must independently reject incompatibility; do not describe this validator as proving exact Qwen tensor compatibility. The already reported actual-PEFT target-module compatibility issue was not duplicated.

Independent probe script: `work/h21a_security_probes.py`; final execution via bundled Python and `rtk proxy` exited 0. No actual trained adapter or quality measurement is attested.

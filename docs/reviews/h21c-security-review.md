# H21c independent security and data-integrity review

Verdict: one P2 bounded-input fix required before training. Initial telemetry concern was repaired during review and independently verified; no stale telemetry finding remains in the final reviewed hash. No source mutations, real ML imports/downloads, full-suite rerun or GPU execution by reviewer.

Scope: AGENTS.md, CLAUDE.md, new semantic loader/trainer/tests, H21c docs/spec and the existing masking/artifact/checkpoint helpers they invoke. Copied H21b runtime/archive context must be replaced with the reviewed root versions before actual execution, as coordinated.

## P2: baseline hashing allocates unbounded input before the JSON limit

Location: `training/scripts/train_development.py`, `digest()` and first baseline digest comparison in `preflight()` (lines 55 and 113 in reviewed source).

`read_json()` advertises a 16 MiB maximum, but `preflight()` first calls `digest(baseline)`, implemented as `Path.read_bytes()`. A tampered downloaded baseline can therefore allocate arbitrary file-sized memory before its externally expected digest is rejected or its size limit is checked. The trusted external digest does not need to change for this attack; rejection arrives after allocation. This can exhaust the remote worker before training, despite the intended bounded preflight.

Concrete probe: created a sparse baseline of 16 MiB + 1 byte and placed a read spy on only that file. `preflight()` reached `read_bytes()` immediately instead of rejecting its known excessive stat size. The spy raised before allocating the file, so the review performed no large-memory test. A larger file follows the same path.

Minimal fix: enforce the baseline byte budget before hashing and read it once with a bounded read, then hash and parse those verified bytes; alternatively combine a stat/read cap with streaming hash verification. Add a regression proving an oversize baseline is rejected before `read_bytes` or any unbounded allocation. Preserve external digest comparison.

## Confirmed boundaries

- Independently materialized mocked media and reconstructed exactly 116 bilingual authored Mind/Type train rows from 136 total rows. Reliance, held-out trajectories and standalone Mind tests are excluded; private audit state/episode objects are removed by projection. Mind has no image; Type retains content-bound current PNG identity.
- Prefix masking uses actual expanded full/prefix tokens, requires an exact prefix and a nonempty completion under the token limit. Type paths are normalized/contained and media digest, PNG signature/format and 800x600 dimensions are checked; model input contains only public prompt, completion and required Type pixels.
- Baseline preflight requires independently supplied baseline/manifest digests, current runtime bytes, immutable base/revision, settings, frozen source/data, dev allocation, exact complete episode/Mind identities and a recomputed summary. Only development deficits trigger training. Standalone test failures cannot alone trigger intervention.
- Harmless probe verification is externally anchored to source/data/evaluation identities and must identify pretraining mode. The operator remains responsible for genuine immutable public-commit download/inspection; a typed commit string alone is explicitly not treated as proof of a network roundtrip.
- Remote guard/opt-in, clean source commit and exact package checks precede ML loading. Base/model revision and remote-code rejection are fixed. Rank4 language-only 112 trainable tensors/802816 elements are enforced; finite loss/gradients/parameters, nonzero gradient and actual parameter change are checked at every step.
- Atomic checkpoint publication uses existing version/pointer helpers. Local trusted optimizer/RNG pickle is confined to the local checkpoint, while export copies only original PEFT config/safetensors and bounded metadata through the exact archive allowlist. No upload/credential or authority-execution path is introduced. Actual archive persistence and fresh-session model evaluation remain unqualified.

## Telemetry repair verification

During review the author introduced `record_memory`, `record_completed_step` and `finalize_training_result`. A fake processor raising an original `ValueError`, combined with final sampling raising `RuntimeError`, now preserves the original exception and writes a durable blocked result. Missing final memory is null with a bounded exception-class marker; unavailable memory is not fabricated. This closes the earlier finalizer masking concern.

Independent probe: `work/h21c_security_probes.py`, final exit 0. It uses fake modules and synthetic media, not installed ML runtime or weights.

Reviewed hashes:

- `training/development_data.py`: `d9a8f02b2c843276051c0220df7d8811499ed1b1d8316a9878fde32a3e288dfb`
- `training/scripts/train_development.py`: `c1218d11d62b65d0ae6fa22dd7209deaec76bea3043a4d7cd6c836dc451eea75`
- `tests/test_development_training.py`: `28f14857b552d37340cfee802ef9f97fdd6f3f131880e71abe46202128c3d066`

These are source-contract findings and checks, not evidence of actual gradients, training improvement, durable adapters or model quality.

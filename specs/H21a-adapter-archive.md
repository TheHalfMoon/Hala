# H21a adapter archival integrity

Status: Source implemented, CPU-tested and independently reviewed; actual remote persistence/load remains pending.

Purpose: make the H21 small-adapter durable archive prerequisite concrete without training, loading weights, changing old loaders, or claiming an asset exists. Own only training/adapter_archive.py, tools/archive_adapter.py, tests/test_adapter_archive.py and this phase's docs/spec.

Acceptance criteria:

- Exact allowlisted flat directory, immutable pinned base/revision and externally expected source/data/evaluation identities.
- Canonical manifest exact-byte SHA256 is externally pinned; all files have exact lengths and SHA256. Extra files, nesting, duplicate JSON keys, malformed paths, symlinks/junctions, unexpected config and malformed tensor headers/offsets fail closed.
- Small adapter < = 8 MiB, whole directory < = 16 MiB; no pickle, base weights, network, credentials or model imports. Bounded optional metadata only.
- Safetensors-only rank4 language q/v LoRA format with exactly all112 tensors/802816 elements across28 layers and pinned Qwen shapes; offline structure validation explicitly does not substitute for actual remote tensor loading/evaluation. Exact qualified PEFT0.20 producer regex/config bytes must roundtrip; no alternate variants.
- Manifest-only pretraining roundtrip is distinguishable from a trained asset. No overwrites; verification precedes publication.
- Adversarial unit tests red before implementation, then green; integrated CPU suite once. Independent PStack security/correctness and genuine Jev/Graft remain required before committing/publication.

Recorded initial red: test module import failed because training.adapter_archive did not exist. Cycle1 review regressions then reproduced actual producer config rejection, incomplete tensor inventory acceptance and a real junction acceptance with Python3.12 detection API absent (three failures/one error). Repairs enforce exact producer compatibility/inventory and lstat reparse-point detection. Final CPU results are reported by the builder to the conductor. Windows real-symlink creation may require a privilege absent in this account; deterministic mocked component rejection covers the guard locally and the real-symlink case runs in Linux CI. The real Windows junction regression executes locally, including absent Path.is_junction.

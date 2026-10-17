# H21c independent security delta

PASS: previous P2 unbounded baseline allocation is closed; telemetry evidence preservation also passes independent probes. No new actionable security finding in this delta. No repository source edits, full-suite rerun, ML import/download or GPU execution by reviewer.

Reviewed bounded input helpers, baseline preflight and Type media reading. Baseline size is checked before file opening, read is capped at limit+1, and changed/oversized input rejects. Its external digest is computed from the exact same bounded bytes subsequently parsed, avoiding a separate hash/parse read. PNG size is similarly checked before a capped read and digest validation.

Independent probe (`work/h21c_security_probes.py`, exit 0):

- Original sparse 16 MiB+1 baseline now rejects with `ValueError` while both `Path.open` and `Path.read_bytes` are forbidden by spies; neither is reached.
- Sparse 2 MiB+1 Type PNG rejects before opening, reading or processing.
- Valid externally hashed JSON parses successfully from exactly one bounded file read.
- Completed numerical step persists despite sampler failure, with null memory and class-only error marker; private exception message is absent.
- Fake processor failure plus final sampler failure preserves the original `ValueError` and writes durable blocked status, null final memory and explicit telemetry error.
- Exact 116-row bilingual train-only projection and positive baseline identity checks still pass.

Reviewed SHA256:

- `training/development_data.py`: `cc0fa09eedab538ba1ef3b412212fc16229ac9ecf93c97cdc3ff1b30a1b1845a`
- `training/scripts/train_development.py`: `17f0270cb721d343aecb5dcab1d35b3c88fcdf3735e3cccf2e180189bd7790f1`
- `tests/test_development_training.py`: `91633217b1e8ed0a9441fc02b0a883fa00c23db54ed214b3b94f0f94ea1d4e68`

Actual training, numerical gradients/updates, durable artifact retention and fresh-session quality evaluation remain separate unexecuted gates. Integrate the reviewed root runtime/archive implementations before running the pilot.

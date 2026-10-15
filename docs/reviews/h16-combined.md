# H16 independent contract review

Artifact: `work/h16-grain.diff`

SHA-256: `189efa3b3d89d4ff6623394e566ca1b6df5576a54a41a7f6722607a76c13c67e` (computed from artifact bytes).

Scope: immutable diff; H16 spec, PRODUCT/CLAUDE/AGENTS, and existing request/action validation and HTTP transport as necessary context. Fresh-context independent combined judge on inherited session model. Source-only candidate review; no release or model-quality approval.

## Must-fix

- None found in the reviewed artifact.

## Worth-considering

- `hala/inference/remote_qwen.py:49-91`, `tests/test_serving_v2.py:164-183`: CPU tests exercise only loader refusal paths; the successful processor/model loader arguments, remote generation token cap, stopping criteria, busy lock, and lock release on exceptions remain source-inspected rather than executed. Add dependency-injected fake Torch/Transformers tests for those seams before extending runtime claims; genuine remote GPU E2E remains a separate required gate.

## Skip-it / confirmed clean

- Correctness: strict request/media fields, tenant and six binding comparisons, PNG byte/digest/dimension/pixel checks, supported origin/DPI/scale, budget/method ceilings, and eligible target checks precede engine inference. Exact point JSON rejects duplicate/unknown fields, nonfinite values, booleans, floats, and unobserved/ambiguous/protected hits. Action validation is reused after server target mapping.
- Correctness: v2 authenticates through existing transport; deadline/disconnect cancellation retains admission until inference worker exit. Explicit cancellation, default unavailable engine, and unsupported plan/decision fail closed. GPU kernel preemption remains an accurately documented limitation.
- Parsimony: adapter and optional loader are small and explicit; policy reuse and separate v2 decoder avoid changes to the frozen H14 protocol. No unnecessary abstraction or dependency expansion found in this artifact.
- Product: implementation matches bounded unqualified click advice, explicit provenance and verification requirement, unavailable default backend, and separate activation/runtime qualification. v1 mock/readiness behavior and H14 protocol files are untouched in the diff. No invented deployment, training, planning, or decision scope found.
- Product: caller-supplied qualification fields fail strict request parsing; injected CPU engines are labeled fake. Media digest equality is correctly documented as binding submitted bytes, not trusted capture attestation.

## Tests actually run

- Bundled Python: `-m unittest discover -s tests -v` in `work/Hala-serving`: **130 tests passed**, 9.659 seconds, including v1 and v2 transport/contracts.
- Preliminary `-m unittest discover -v` found **0 tests** and exited 1 because discovery lacked the explicit tests start directory; this was not counted as verification.
- SHA-256 calculation on the immutable diff matched the assigned artifact hash.
- No GPU, model loader success path, model weights, remote serving/E2E, Jev, Graft, or hosted CI execution performed by this judge. No source edits or commits.

panel: full component — correctness ✓ fresh-context Codex inline bar · parsimony ✓ inline bar (no ponytail provider invoked) · product ✓ spec/source/CPU contracts (no UI surface) · security — separate judge required and outside this component

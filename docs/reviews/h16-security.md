# H16 independent PStack security judge

Verdict: no confirmed must-fix security findings in this source candidate.

Immutable reviewed artifact: `work/h16-grain.diff` SHA-256 `189efa3b3d89d4ff6623394e566ca1b6df5576a54a41a7f6722607a76c13c67e` (independently recomputed with Python hashlib). Reviewed implementation: `work/Hala-serving`; context: H16 spec, CLAUDE.md and directly relevant existing policy/transport validators. Fresh-context session-model security judge; no d1 verdict used.

Findings:

- must-fix: none confirmed.
- worth-considering: `hala/inference/remote_qwen.py:52` — explicitly pass `use_safetensors=True` to the model loader to enforce tensor-only artifact selection instead of relying on Transformers' preference and the pinned repository contents; optional deserialization hardening, not evidence of an exploitable path in the pinned candidate.
- skip-it: `hala/inference/vision_candidate.py:52` / candidate documentation — submitted digest and duplicate identity fields establish consistency, not capture authenticity or freshness; this is explicitly documented, proposals confer no execution authority, and the authenticated caller supplies policy. Future native integration must provide trusted observations; do not invent a capture attestation mechanism in this grain.
- skip-it: `hala/inference/remote_qwen.py:32` — environment markers are not adversarial provider attestation, as documented; the loader is a server-owned, opted-in seam rather than an HTTP caller-controlled loading path. Remote session/free-GPU eligibility still requires separate operator verification.

Actual validation:

- `rtk proxy C:/Users/Shehr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m unittest tests.test_serving_v2 tests.test_serving -v` from `work/Hala-serving`: 37 tests passed in 2.786 seconds. Includes authentication, cross-tenant checks, tamper/identity, protected/redacted/ambiguous targets, strict JSON output, default unqualification, local/Kaggle/cache guards, worker admission retention, deadline and disconnect cancellation.
- Independent executable probes: `work/h16-security-probes.py`, run with the same bundled Python via `rtk proxy`: five passed. Nonblack redaction away from the selected target, invisible RGBA content retaining nonblack RGB, and compressed PNG text over Pillow's decompression limit each returned `invalid_request` with zero engine calls. Additional authority field in engine provenance and protected target overlap at the selected pixel each returned `invalid_backend_output`.
- Initial probe invocation failed because the external script directory did not import the checkout; added the checkout to the probe's module search path and reran successfully. No implementation files changed.

Confirmed clean: new HTTP paths inherit bearer authentication before body/model processing; tenant mismatch rejects; all media bindings match validated bounded IDs; byte/pixel/digest/dimension checks precede inference; actual redaction pixels are checked; protected policy and target eligibility survive output mapping; exact response fields reject model authority/confidence additions; cancellation/deadline checks prevent completed inference from emitting stale advice; capacity stays held until worker exit. No caller-triggered model loading, arbitrary model/cache selector, shell execution, pickle request deserialization, public bind, readiness qualification or success receipt was introduced.

Limits: CPU contract/fake-engine evidence only. No weights downloaded, loader invocation, GPU inference, provider attestation, native execution, real model quality claim, release approval or independent Jev/Graft/CI substitution. Deadline cannot preempt a decoder operation or an in-progress GPU kernel; existing retained admission bounds concurrent workers, and remote behavior remains unqualified.

panel: full — security ✓ independent fresh-context PStack inline bar (session model); other bars owned by the parent review.

# H2–H11 independent security review

- Role: fresh-context PStack security judge, session model; inline security bar.
- Snapshot: `work/h2-h11-review.diff`, SHA-256 `48b1f007da4af417eb072ad391734f9dfbf15e6775c0ac6014df4f5777b80b8b`.
- Scope: added contract, inference and serving modules; `tests/test_contracts.py` and `tests/test_serving.py`. All 11 inspected source/test files matched the immutable snapshot after normalizing line endings.
- Disposition: **no must-fix security finding identified within this bounded mock scope**.

## Findings as data

```json
{"must_fix": [], "worth_considering": []}
```

## Clean security bars

| Bar | Evidence | Result |
| --- | --- | --- |
| Auth on all HTTP paths; tenant bound to fixed credential | `hala/serving/__init__.py:83` uses constant-time bearer comparison; `:98` enforces configured tenant; `:245` authenticates before route dispatch. All seven unauthenticated route probes returned 401; cross-tenant suite test returned 403. | Clean |
| Untrusted output cannot gain authority | `hala/serving/__init__.py:110` reparses backend output against its request. `hala/contracts/proposal.py:132` checks all six bindings; `:147` checks action vocabulary/ceiling/targets. `hala/contracts/responses.py:118` requires advisory-only decisions. Independent binding mutations: 18/18 rejected across plan/action/decision. | Clean |
| Structural validation fails closed | Exact object fields, bounded arrays/strings, finite numeric values, duplicate JSON rejection, no coercion. 416 nested request substitutions: 397 contract rejections, 19 accepted substitutions, zero unexpected parser exceptions. Duplicate case-insensitive auth, folded header, unexpected GET body and Expect probes returned 400. | Clean |
| Observation and action containment | `hala/contracts/proposal.py:139` rejects unobserved/protected/redacted targets; `:156` checks click frame, coordinate space, scale and containment. The focused suite covers forged targets, stale bindings, DPI mismatch and method escalation. | Clean |
| Bounded service resources | `hala/serving/__init__.py:345` admits before spawning; shared lease holds capacity until handler and worker finish. Focused tests verify absolute read deadlines, trickle headers, body/header/output limits, cancellation and continued capacity exhaustion while an expired worker is still running. | Clean |
| Secrets/error details and qualification claims | Fixed public error messages; token excluded from Config repr; no request logging path observed. `hala/contracts/proposal.py:40` rejects qualified mock metadata; `hala/inference/__init__.py:71` makes real backend unready and unavailable. | Clean |

## Verification

- Bundled Python: `python -m unittest tests.test_contracts tests.test_serving -v` — **45 passed**.
- Independent probe: `work/security_boundary_probe.py` — 416 request substitutions, 18 response binding mutations, 11 raw HTTP probes; all expected results obtained. The probe is outside product source/tests.
- No source changes made by this judge.

## Limits

- This is an independent PStack inline review, **not a managed Alibaba, d1, Jev, or Codex Security release gate**.
- Inference cancellation is cooperative. A future backend that ignores cancellation can retain a bounded slot after a response timeout; the present implementation correctly avoids releasing that slot early. This review does not qualify such a backend.
- Binding checks tie advice to supplied request identifiers and observation generation; live device freshness, authenticity of observations, execution authorization and final-state verification remain Aladdin/local-policy responsibilities. No local executor is present in the reviewed surface.
- CPU mock transport and synthetic fixtures only. No model weights, native desktop actions, training efficacy or model qualification were tested or inferred.
- Aladdin was not edited or independently qualified. Reference `0c3c86c8acfb57e3dd3ce83c83ee8cfd3b5035c1` is a proposed compatibility reference.

panel: security ✓ independent fresh-context inline judge (session model); managed external security/release gates not run by this judge.

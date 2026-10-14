# H11/H12 inference and integration

## Goal
Expose unified Hala responsibility endpoints through an authenticated bounded CPU mock and a fail-closed real-backend qualification seam.

## Coordination
Depends on: H2
Surface: hala/serving, hala/inference, tests/test_serving.py, Dockerfile
Review: full

## Requirements
/health, /ready, /models, /v1/hala/plan, /v1/hala/action, /v1/hala/decision. Explicit API version, constant-time bearer validation, bounded payload/body/output, strict JSON, typed error responses, request deadlines, concurrency limits, cancellation and tenant isolation. Readiness accurately describes mock mode; real backend fails until separately qualified. No arbitrary code execution, remote URL fetching or secret logging.

## Acceptance criteria
- In-process and real loopback HTTP tests demonstrate successful mock paths and auth failures.
- Duplicate keys, malformed JSON, excessive bodies, unauthorized methods, concurrency, timeouts and cancelled work fail closed.
- Metadata distinguishes mock from real checkpoints and readiness remains false for unqualified real backend.
- Docker and setup instructions reproduce CPU service with no weights or external APIs.
- Compatibility register identifies proposed Aladdin contract revision and gaps; service never returns execution receipts.

## Hardstop
No production deployment, real weights locally, invented approvals or unqualified native executor integration.

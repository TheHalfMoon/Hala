# Hala v1 mock service and Aladdin compatibility

This is an implemented development transport for deterministic CPU mocks. It does not load a checkpoint, execute computer actions, grant approval, or produce execution receipts. Real model and native Aladdin integration qualification remain blocked.

## Compatibility register

| Reference | Immutable revision | Status and v1 adapter requirement |
|---|---|---|
| TheHalfMoon/Aladdin AI architecture and proposed model contracts | `0c3c86c8acfb57e3dd3ce83c83ee8cfd3b5035c1` | Proposed reference only. No adoption or live executor interoperability claimed. |
| Hala request/proposal contracts | `hala.request.v1`, `hala.proposal.v1` | Implemented independently; the Aladdin adapter must carry task and device session binding as well as request, tenant, device, and observation generation. |

Hala outputs are proposals under Aladdin's allowed vocabulary and method ceiling. The proposed-reference Aladdin payload is insufficient unless its adapter supplies the required `task.task_id`, `device.session_id`, and other v1 bindings from trusted local state. The adapter must preserve these values, reject stale generation/session/tenant responses, and route every proposal through local policy and independent final-state verification. A remote response cannot mint approvals, leases, or receipts. No Aladdin repository was modified. A production adapter is not implemented or qualified by these mock tests.

## Start locally

Python 3.11 or newer and the standard library suffice. From the repository root:

```powershell
# Generate a development secret in memory; never put it in source or a command argument.
$env:HALA_BEARER_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:HALA_TENANT_ID = "tenant-1"
$env:HALA_BACKEND = "mock"
python -m hala.serving
```

The service binds only `127.0.0.1:8765`; `HALA_PORT` can choose another loopback port. A missing or invalid token/tenant/port fails startup with a fixed error. Each server instance binds one credential to one tenant. For isolated development tenants use separate instances and credentials. The service does not trust a caller-selected tenant header. Multi-tenant identity issuance and production TLS infrastructure remain future qualification work.

All endpoints, including health and readiness, require `Authorization: Bearer <token>`. The service does not log request bodies, credentials, or interpreter exception details. Responses have `api_version: "hala.v1"`, `Cache-Control: no-store`, and a closed connection. UTF-8 JSON bodies require `Content-Type: application/json` and a single decimal `Content-Length`; chunked encoding, duplicate headers, duplicate JSON keys, nonfinite numbers, malformed UTF-8, and pipelined requests are rejected. The optional `; charset=utf-8` content-type form is accepted.

| Endpoint | Method | Result |
|---|---|---|
| `/health` | GET | Liveness plus explicit backend metadata |
| `/ready` | GET | 200 only when current backend can serve; mock is ready but `qualified=false`; unqualified real is 503 |
| `/models` | GET | Mock/checkpoint/qualification metadata; no real weights implied |
| `/v1/hala/plan` | POST | Validated HalaMind plan |
| `/v1/hala/action` | POST | Contextually validated HalaType proposal |
| `/v1/hala/decision` | POST | Contextually validated HalaReliance decision |

For a complete valid request, see `tests/test_serving.py:payload`. Important binding fields are:

```json
{"schema":"hala.request.v1","request_id":"r-1","tenant_id":"tenant-1",
 "task":{"task_id":"task-1","goal":"Click Save","step_index":0,"max_steps":10},
 "device":{"device_id":"device-1","session_id":"session-1","os":"windows-11","locale":"en-US"}}
```

This excerpt is not a full request: `observation`, `allowed_actions`, and `method_ceiling` are also required. `cancelled` is optional and defaults to false. See `examples/hala-request-v1.json` and `examples/hala-proposal-v1.json` for complete validated synthetic payloads. No remote URL is fetched and no pixel inference is claimed. POST responses wrap the contract record in `{"api_version":"hala.v1","result":...}`. Errors use `{"api_version":"hala.v1","error":{"code":...,"message":...}}` with fixed public messages.

## Limits, deadlines, and cancellation

Defaults are 4 admitted connections, 8 KiB total request line/headers, 64 KiB body, 64 KiB response, and an absolute 2 second request deadline measured at accept. `Config` can choose stricter bounds and rejects zero, negative, nonfinite, boolean, fractional integer limits, and excessive values. Concurrency admission occurs before a handler thread is spawned. A full service immediately returns 503 using a bounded write; the OS listen backlog is 16. Every connection serves exactly one request and closes.

Reading headers/body, waiting for inference, and sending a success response share the absolute deadline. A timeout returns 408 with a short bounded error-write grace period. A pre-cancelled request returns 409. A client disconnect or unexpected additional bytes while inference runs sets the request's cancellation event and discards its output. Individual requests hold immutable contract records and independent deadline/cancellation state; no output cache is shared across tenants.

Python threads cannot forcibly terminate an uncooperative backend. The development mock completes promptly and checks cancellation. Future runtimes must check the supplied `RequestContext` or provide separate process/runtime cancellation. Crucially, a timed-out/disconnected worker continues holding its admission slot until it actually exits; repeated cancellation cannot create unbounded backend workers. An uncooperative worker may exhaust availability, and bounded process isolation must be qualified before a real backend is deployed.

Setting `HALA_BACKEND=real` exposes a fail-closed seam: readiness is false and inference returns `backend_unqualified`. There is no caller-controlled qualification flag. Installing a real backend requires separate checkpoint, commercial license, runtime, resource, security, and benchmark qualification.

## Offline checks and development container

```powershell
python -m unittest discover -s tests -p "test_*.py"
docker build -t hala-mock .
docker run --rm --network none --name hala-mock --env HALA_BEARER_TOKEN --env HALA_TENANT_ID hala-mock
```

The Docker recipe uses Python 3.11.14 with observed Docker Hub digest `sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d`, runs offline transport tests, and serves as an unprivileged user. The actual Linux build passed 20 transport tests. A separate run with no external network, read-only filesystem, all capabilities dropped and no-new-privileges also passed 20 tests. It has no model/package downloads beyond the base image and publishes no host port. Loopback stays inside the container; authenticated smoke calls can run through `docker exec` in that container. Cross-container exposure, TLS, native Aladdin interoperability and production deployment remain unqualified.

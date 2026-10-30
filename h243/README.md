# H243 — Browser-context offline defense for research-only GUI capture

## Confirmed gap

The existing H231, H236, and H242 Playwright capture paths blocked HTTP
requests with `context.route("**/*", ...)` and disabled service workers.
Those measures alone do **not** cover every Dedicated Worker WebSocket.
This is a capture-isolation defect, not evidence of actual owner-data leakage.

## Source-only fix

All three paths now call `h243.browser_offline.require_browser_offline(context)`
immediately after creating each browser context and **before** opening a page,
loading a local file, or injecting original donor JavaScript. The helper
requires Playwright's `context.set_offline(True)` and fails closed if the
method is absent or raises. Existing request routes and service-worker
blocking remain in place. No fallback to route-only isolation is allowed.

## Owner Windows real browser evidence — 2026-10-30

An opt-in, self-authored, `file://` + ephemeral **127.0.0.1-only**
WebSocket probe tested both page-script and Dedicated Worker attempts using
the already-installed Microsoft Edge browser:

- No offline guard: **2 loopback TCP connections accepted**.
- `set_offline(True)` before navigation: **0 connections accepted**.
- Result: `H243_LOOPBACK_NATIVE_PASS research_only=true`.

Reproduce without owner images, model weights, external hosts, or downloads:

```powershell
python -m h243.loopback_probe --native
```

The probe creates a temporary authored HTML file outside Git and removes it.
Its baseline makes a local loopback connection only. It neither reads owner
data nor tests or claims real model behavior.

## Qualification and scientific boundaries

- H231/H236/H242/H243 targeted synthetic suite on owner Windows:
  **40 passed, 23 subtests passed**.
- H243 synthetic suite under Python `-O`: **5 passed, 3 subtests passed**.
- The native probe proves only the tested page/worker WebSocket paths.
  **It does not prove OS-level network isolation**, WebRTC/UDP containment,
  browser extension isolation, or absence of future browser escape routes.
  Keep all capture output owner-private and research-only. Do not admit
  TRAIN, DEV, or OOD based on this guard.
- H199's admitted TRAIN manifest allowlist remains EMPTY. H212 research
  scenes remain optimizer-prohibited by H222. H173's scored 32-case OOD
  remains permanently consumed and sealed. No neural model calls,
  optimizer steps, accuracy improvement, paid compute, merge, or release.

Public GitHub changes consist only of this generic source code, synthetic
tests, and documentation. No private screenshots, labels, hashes, source
identities, gold, or local receipts are included.

# H39 — independent fresh-capture consistency review (CPU-only)

**Scope:** Hala experimental proposal validation only. No native click, no endpoint, no process execution, no model inference, no new GPU, no host permission grant, and no release qualification. Stack on [H38 Draft PR #21](https://github.com/TheHalfMoon/Hala/pull/21) at signed head `631e37ed60598d8f5b8943e62d8a03d5a39ac00d`.

## Problem

H38 rejects a visually grounded proposal if its selected accessibility label is not the unique label explicitly specified by an English/Arabic user request. H29 already requires unambiguous geometry against the full observed tree. But a model response can be correct for the **previous** screen while the user, window, UI, or device/session has changed during inference. Historic H23 synthetic `apply_proposal` checked binding against the original request payload, not a newly captured independent screen.

## New experimental primitive

`hala/inference/fresh_snapshot_review.py::review_same_fresh_snapshot(original, proposal, recaptured, trusted_goal=...)` accepts a **separately host-supplied** fresh capture plus a goal presumed to originate from the trusted user/host. It validates both image-request payloads through the current strict v2 parser and requires a second self-consistent PNG/identity binding. It refuses:
- Any change in full validated task/request, device/session, frame generation, window reference/title/process/protection, target list/names/geometry, task step/budget/cancellation/policy, redaction/locale or truncation.
- Any difference in the original vs recaptured model-visible PNG and bound media fields or a mismatch between the host-supplied trusted goal and the original/fresh task.
- Any malformed, contradictory, qualifying or authorization-claiming model proposal; any altered request ID / tenant / task / device / session / frame / image digest; any click outside the single observed target.
- Any ambiguous/partial accessibility tree or non-unique literal label compared against all observed names, in both English and Arabic.

On success, the function deliberately returns **`None`**, not an action, signed permit, authenticated identity or authorization token. It makes **no UI interactions**. It is a *necessary consistency check in a future system, not a sufficient condition for performing an action*.

## Engineering and safety limits

- No method in Hala currently ensures `recaptured` was collected after the model responded or originates from an authenticated, non-user-forgeable host; similarly `trusted_goal` has no origin proof in this CPU function. If the calling application supplies the same prior object or fabricates both records, a matching result is meaningless.
- Even a genuine second capture is subject to a **time-of-check/time-of-use race**; the host needs its own atomic window/session lock, authorization, new capture ownership, execution gating, risk controls and independent postconditions before interacting with an application. Reusing frame identifiers or pixel hashes without authenticated, fresh capture is insufficient.
- This exact-label grammar is deliberately narrow, and cannot verify free-form natural language or inaccessible/unlabeled controls. Arabic expert/adversarial RTL tests and independently collected real application screens remain future evidence requirements.
- The new module is **not wired to any native executor**; H38's opt-in gate remains default OFF and `execution_authorized=false`. No claim of improved HalaType neural accuracy: genuine H37 model target hits remain 32/64 on publicly authored browser GUI fixtures.

## Regressions

`tests/test_h39_fresh_snapshot_review.py` uses experimental Service/CandidateBackend with fake CPU engine, never native actions. Cases include a valid matching pair returning no permit; an actual uniquely wrong target refusal; stale generation or session changed with **all fresh media binding fields updated self-consistently**; different PNG pixels with correct new digest; unchanged pixels but changed accessibility name; changes in window, task, locale, target bounds, protection, redaction, policy, truncated tree and user goal; and tampering with proposed identity, authorization flags, point type, geometry and confidence. This is **not H23 actual model CUDA lifecycle evidence**.

Run offline:

```powershell
python -m unittest tests.test_h39_fresh_snapshot_review -v
python -m unittest discover -s tests
python -m compileall -q hala/inference/fresh_snapshot_review.py tests/test_h39_fresh_snapshot_review.py
```

Source-only gates for local Python, Graft wiring, signed GitHub head CI, and independent security reviewer status must be reported with separate evidence. Do not merge or release solely from these tests.

## Verified Windows local qualification (2026-10-20)

- H39 focused **10/10** tests passed on actual candidate-shaped CPU fake-engine proposals. Includes self-consistent fresh generation/session changes, different valid PNG with corrected digest, and changed observed accessible labels with unchanged PNG.
- First full Hala suite concurrently with Graft: **530 tests, 3 failures / 2 skipped**. A visible lifecycle fault test expected `busy` but received `backend_unqualified` under concurrent load; standalone repeat of that exact test **passed**. Three first-pass failures were observed, but complete individual traces of all three were not retained. Do not mark this initial run successful; resource/timing sensitivity remains unresolved.
- Repeat **without simultaneous Graft**: `python -m unittest discover -s tests -q` **530 run / 528 passed / 2 skipped**, exit 0, `compileall` exit 0. The unsanitized local temporary test transcript was screened for simple credential-like patterns (zero matches) and retained in `reviews/h39-full-suite-20261020.log`; SHA256 **`286564cbe4f6c8babffb32ac0b550aed4327c8281e571dbf6ba0054d2e46daaa`**. This is one green complete run, not proof that concurrency timing failures are fixed.
- Genuine Graft **wiring-only** build/check PASS **1,633 nodes / 4,404 edges**. Deep semantic meaning tier is **not run**, and independent PStack/D1/Alibaba model review is not claimed.
- No Kaggle or GPU was used, no weights loaded, no paid service and no native host actions occurred.

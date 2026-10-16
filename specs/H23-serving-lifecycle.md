# H23: private remote serving lifecycle experiment

Status: source CPU-tested and independently gated; actual remote evidence pending. Depends on: reviewed H16 transport, H21f evaluation admission, H22 source base. Surfaces: new optional operator modules, fixtures/config, CPU tests; no existing public API or backend defaults change. Review: full.

Use the existing authenticated v2 click-proposal loopback service with a single pinned Qwen worker process. The fixed child module uses bounded JSON pipes, no pickle or caller program paths. Deadline, cancellation and worker failure destroy only the owned child and fail closed; explicit restart loads a fresh model. Operator faults are private Python operations, never HTTP fields. Cache admission reuses reviewed evaluation guard with a fixed ephemeral cache. No local weights or generic Linux fallback.

New rights-cleared synthetic workflows: two independent three-step open/toggle/save tasks, English and Arabic, causal observations and separate persisted-state verifier. No H19/H21 fixture, prompt, verifier or training changes. All proposals remain untrusted; only the synthetic state machine applies them. No native execution, receipts, production adoption or general quality claims.

Acceptance:
- [x] CPU loopback sequential turns with independent task outcomes and unsafe proposal rejection.
- [x] Owned worker lifecycle CPU tests: cancellation after CPU work starts, absolute deadline, failure/restart, output/injected-resource bounds and exact revision check.
- [x] Source remote-only admission, inherited strict auth/media/policy compatibility and immutable source/config/fixture freeze; no actual remote qualification inferred.
- [ ] Optional actual CUDA start evidence and bounded allocator-budget exhaustion probe; CPU fakes explicitly excluded from actual CUDA/OOM claims.
- [ ] Full CPU suite, independent combined/security, genuine Jev/Graft and prospective remote run remain separate gates.

Assumptions (assumed): single worker, no queue, explicit operator restart, 60-second request ceiling inherited from transport. Actual cancellation qualification requires a CUDA event recorded after a real active-model matrix operation, enqueued before exact transport cancellation receipt and still pending at the later private child acknowledgment, followed by owned-child reap. CPU timers, pre-enqueue markers and completed-forward timestamps are insufficient. Actual remote evaluation, resource exhaustion and worker restart remain pending until root operator evidence. Torch 2.11.0, Transformers 5.16.1, exact model/processor 89644892e4d85e24eaac8bacfd4f463576704203; frozen font bytes, Pillow and RAQM checked before every load/restart. The full first-party Python superset is frozen and exact external source commit/clean Git checkout required before ML.

Initial source gate: 248 CPU tests passed, one skipped (24.903s). Review repair scoped suite: 16 CPU tests passed after four initial regression failures; compileall passed. Prospective 53-file freeze SHA256 `20b57ebb7e66b59f480cb89cb7fd363e671d212d48249ec99356856bc16730ae`. Independent review deltas/Jev/Graft/signing and actual evaluation are deliberately not claimed by the builder. Incremental atomic task/fault/load progress retains evidence if later probes or reloads fail.

Final repair full-suite gate: 253 tests run, one skipped, passed in 27.987s. No actual CUDA/allocator qualification or adoption claimed.

Second repair cycle: owned handle remains until a successful wait confirms reap, double wait timeout records explicit unreaped/error state, restart is prohibited while owned/unreaped, and a later bounded close can retry. Buffered pipe closure waits for reap; control cleanup/reader join stay bounded. Red-to-green double-timeout/retry regression; scoped 17 tests and final full suite 254 tests run, one skipped, passed in 28.066s. Current freeze superseding earlier hashes: `063dde2eee1635ce1a52e40748ef76c06fb63a2fedb8a69da3322e269261015c`. All actual qualification flags remain false.

Third repair cycle: retained ownership cannot serve inference after a fault/cleanup. Explicit unusable state rejects pipe writes/queue reads until confirmed reap and an explicit successful fresh-worker provenance handshake. Red-to-green queued stale-output regression; scoped 18 tests and final full suite 255 tests run, one skipped, passed in 25.905s. Current freeze superseding all earlier hashes: `4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a`. Third-cycle cap reached; root final independent delta review must pass or leave the phase parked. Actual qualification remains false.

Final source disposition: independent combined cycle-three and security delta reviews pass with no remaining must-fix. Genuine root Jev and Graft execution is archived with all initial/delta findings in `docs/reviews/h23-source-gates.json`. Jev correctness-concern probabilities 0.58/0.63/0.59 remain advisory; Graft confirms graph freshness/impact only, with deep context false. These source gates do not qualify actual CUDA, model outcomes, native integration or production.

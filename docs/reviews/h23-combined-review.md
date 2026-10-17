# H23 independent combined source review

Scope: lifecycle branch against `733fc7e11cfc147a463d388ee65559acd1fa847b`; correctness, parsimony and product. Read the new source, fixtures/config/freeze, CPU tests, H23 spec, ADR and product/conventions. No source edits, model download or GPU execution.

## Must-fix

- **P2 — `evaluation/serving_lifecycle.py:188-196`, `hala/serving/lifecycle_worker.py:58-72`: the fault probe cannot establish cancellation during active CUDA.** It triggers disconnect only after a synchronized forward has completed. The second forward's entry event is emitted before that forward enqueues its model work; absence of its completion event after process termination does not establish that CUDA was active when cancellation was received. Generation can also finish before the disconnect. This misses the user's explicit active-CUDA product requirement, even though all qualification flags correctly stay false. Add bounded private instrumentation that observes an actual pending CUDA event enqueued after a real model operation and records its pending/completed state at cancellation receipt, then terminates/reaps the same owned child. A separate private control/ack path or monitor can provide this without public HTTP fault fields; pre-enqueue entry markers, CPU delays and toy CUDA work are insufficient. Add meaningful CPU contract tests for the evidence/cleanup path; actual overlap qualification still requires independently inspected remote traces.

## Worth considering

- `evaluation/serving_lifecycle.py:163-219,260-267`: preserve partial fault evidence on failure. `probe_lifecycle` accumulates results locally and returns only after all modes and restarts succeed. If a forward is not observed, cleanup times out, or a later reload fails, `result.json` retains only the exception type and loses the allocator result, earlier modes and failing mode's trace/fault. Store/update the lifecycle record incrementally or copy bounded diagnostics in a failure path so a failed expensive remote run remains inspectable.

## Skip-it / clean

- Owned fixed-module subprocess, bounded JSON messages, explicit restart, fail-closed worker errors and inherited authenticated v2 validation fit the private experiment; no native execution or default public backend changes found.
- Synthetic open/enable/save states are causal, each task persists separately, and the separate verifier reads persisted state rather than trusting service claims. Their deliberately small/fixed layouts cannot establish general quality, and the implementation does not claim that.
- The allocator operation requests a bounded 8 MiB under a per-process budget and restores the prior allocator fraction. It is described as allocator-budget exhaustion, not whole-device physical OOM.
- No unnecessary dependency or speculative public abstraction found. Dense source formatting is not a blocking issue.
- All CUDA/native/overall qualification flags remain false. CPU fixtures are clearly labeled and excluded from actual CUDA/OOM evidence.

Validation: `rtk python -m unittest tests.test_serving_lifecycle -v` — 11 tests passed in 5.043 s. This exercises CPU child process and loopback contracts only. No actual CUDA overlap or allocator qualification performed.

panel: full contribution — correctness ✓ fresh-context Codex combined judge · parsimony ✓ combined judge · product ✓ combined judge · security separate/root-owned · Jev/Graft root-owned

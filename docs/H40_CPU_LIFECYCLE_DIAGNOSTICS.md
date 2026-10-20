# H40 — bounded CPU-child lifecycle diagnosis, not a CUDA qualification

**Status:** reproducible source-controlled diagnostics, CPU-only, no fixes asserted to the H23 GPU lifecycle, no real model execution or host action; stack atop [H39 PR #22](https://github.com/TheHalfMoon/Hala/pull/22) signed source `15017fcfd7c8e4155959c5db609982e1f2ec46df`.

## Trigger / honest unresolved issue

The H39 full Windows suite first ran **concurrently with a separate Graft build/check**. Its 530 tests recorded **three failures** (2 skips); one visible `test_deadline_failure_oversize_and_resource_fail_closed` mismatch at the `resource` subtest expected `busy` but observed `backend_unqualified`. All failure traces were **not** saved, so the three cannot be fully localized or attributed to CPU load. The specific lifecycle test passed isolated, and the later serial 530-test suite also passed (528/530 with 2 skips) in 115 seconds. This is **not** proof of a bug fix. A load-dependent race, transient scheduling, or test fixture artifact remains possible.

## Measured follow-up (Windows 11, Python 3.14.7)

H40 introduces create-only, strictly bounded, actual owned-subprocess CPU lifecycle fault diagnostics. The worker `tests.lifecycle_child` is a benign fixture. WorkerEngine starts only this subprocess through its exact owned Popen handle and reaps it; no real model, no GPU, no network, no UI, no paid API. Each trial exercises one normal response and four fault responses with **no retry** (deadline, crash, oversized output, resource busy); every attempt and exact expected/observed code, elapsed milliseconds and child-reap state is fsync-written to a dedicated JSONL journal.

| Trial | Runtime | Observed | Result |
|---|---:|---:|---|
| Eight consecutive rounds, no added CPU threads | 8.93 s | **40/40 exact status/ownership outcomes** | PASS |
| Eight consecutive rounds, two bounded internal CPU contention threads | 9.83 s | **40/40 exact status/ownership outcomes** | PASS |
| Total across distinct trials | 18.76 s | **80/80 fixture observations** | PASS, does not reproduce the earlier suite flake |

The trials use a 2-second non-wait context deadline and a deliberately short 100 ms deadline for `wait` cancellation. Outcomes are stored in `reviews/h40-lifecycle-isolated-20261020.{json,jsonl}` and `reviews/h40-lifecycle-loaded-20261020.{json,jsonl}`. Data from both runs is **not H23 genuine mid-generation CUDA cancellation or worker reload qualification**. They cannot prove the earlier full-suite failures were caused by parallel Graft: this trial was not the same concurrent workload nor identical timing.

### Evidence integrity

`evaluation/h40_cpu_lifecycle_stress.py::verify_receipts` checks immutable journal SHA256, complete trial count, schema, expected ordering and exact fault/status/reaped semantics. Rejects reordered, truncated, missing, modified, incomplete or inflated `passed` rows, and false claims of GPU/native execution. `tests/test_h40_cpu_lifecycle_evidence.py` validates both actual retained receipts, tamper detection, create-only output, input bounds and refusal to overwrite. The owner hashes the benign fixture source `tests/lifecycle_child.py` in its run metadata. These are local machine observations, **not independent signed hardware attestations**.

### Source and run

```powershell
python -m evaluation.h40_cpu_lifecycle_stress --output <new-output> --rounds 8
python -m evaluation.h40_cpu_load_trial --output <other-new-output> --rounds 8 --workers 2
python -m unittest tests.test_h40_cpu_lifecycle_evidence -v
python -m unittest discover -s tests -q
```

Use only a **new output path** for another run; never overwrite a saved receipt. Do not increase runtime or CPU worker counts without intentional scope review. Source DIFF only adds a controlled diagnostic/receipt verifier and tests; production `WorkerEngine` runtime, timeout values, H37 neural predictions and H39 host reviewer are unchanged.

## Remaining actions and blockers

- Reproduce and retain **all** failure traces under the exact production-representative concurrent workload before proposing a runtime or fixture timing fix. Avoid interpreting isolated 80/80 success as resolving a concurrency issue.
- Review possible scheduler starvation, fake-child startup and deadline ownership under CI Python 3.11/3.12 as well as Windows 3.14.7. Current data support only the named Windows observations.
- H23 still requires authentic provider CUDA model lifecycle, cancellation receipt, worker-failure restart, teardown and reload under the version-pinned Linux GPU environment; independent external UI, Arabic review, trusted host action authority and release gates remain open.
- Keep work Draft/unmerged unless governance allows; no native execution or production qualification.

## Actual Windows evidence and source gates — 2026-10-20

- Isolated receipt JSONL SHA256 `622c1f967bdbabb96acecf85c273c7868aa584f8e82af03e6c26853bd9b39f30`, envelope SHA256 `4e0d11b520725e81af39ef8302443ea883046ffff8d0f9819271f632b9ee7714`. Exactly 40 complete records and eight rounds.
- Controlled CPU-thread contention receipt JSONL SHA256 `dc56315b4950192d4ead5006d65b3b6dd37057dbc0883abdd1733a525494b419`, envelope SHA256 `539f237b9e792b418a86794e22ea4a9b699b9fe70939b38923021d15b16dfe95`. Exactly 40 complete records and eight rounds. The returned CLI report named two load threads; the persisted envelope does not independently attest their activity.
- Verified through `verify_receipts` and the **3/3 H40 focused CPU regression tests**, including a forged `last_fault_code` with recomputed journal digest that must still fail, truncated/mutated bytes, false `native_actions` and inflated `passed`. No retries, no CUDA.
- Full **serial** Windows test suite **533 tests run, 531 passed, 2 skipped**; exit 0; **93.596 seconds**; `python -m compileall` PASS. Actual full-suite log SHA256 `69c75d2ef7574c506ec6923f68cc447764ce2a8b3aada26a8da3f8c8a4ad3f96`, retained in `reviews/h40-full-suite-20261020.log`. A later focused-only change strengthened receipt semantic validation and passed focused 3/3; exact-head CI must verify all tests against final committed code.
- Genuine Graft **wiring-only** build/check PASS **1,645 nodes / 4,450 edges**; run again **after** the final one-line receipt semantic validation patch, and `graft check .` confirmed the graph is in sync. Deep meaning tier not run.
- Independent model-driven local D1, PStack, Alibaba Code Review and Jev analysis were **not** executed/claimed here; a passing CPU stress receipt is not an independent security approval.

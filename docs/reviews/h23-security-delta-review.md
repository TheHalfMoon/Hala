# H23 independent security delta review

Reviewed repair against the earlier H23 review, builder handoff and final prospective freeze `20b57ebb7e66b59f480cb89cb7fd363e671d212d48249ec99356856bc16730ae` in `work/Hala-lifecycle`.

Verdict: no remaining must-fix security findings in the reviewed source delta. Both earlier P2 findings are resolved.

- **Resolved — admission ordering:** `evaluation/serving_lifecycle.py:47` checks provider/cache, source identity, Git cleanliness and font/Pillow/RAQM before initial load. `restart_worker()` gates every explicit restart; parent `_spawn()` and child `load()` repeat preflight before process creation or ML imports/model download, respectively.
- **Resolved — source closure:** `source_files()` covers the 53-file first-party Python/config/test superset, including both previously missing modules. `verify_sources()` checks exact membership and bytes; `verify_git()` checks the externally supplied exact commit and complete tracked/untracked cleanliness. Independently verified the final freeze and observed the current uncommitted checkout correctly rejected as dirty.
- **Clean — new cancellation evidence channel:** inherited `socketpair` and fixed child module retain owned-process scope; no public HTTP control surface, caller executable, arbitrary PID or shell added. Parent acknowledgment input is bounded to 4 KiB with a 250 ms absolute deadline. Event history is bounded to 128 records. Child query selects an event recorded before the actual transport cancellation timestamp; event provenance comes from real active-model CUDA matrix operations. Failed/missing acknowledgment cannot avoid child termination. Actual overlap remains independently assessed and all qualification flags remain false.
- **Clean — failure evidence/cleanup:** result progress uses the inherited atomic persist helper before risky operations and after outcomes; task attempts, raw fault traces/acknowledgment and reload errors survive partial failure. Fault server/socket/model cleanup is protected by `finally`, including when progress persistence raises. Loopback bearer authentication, strict v2 policy/media validation and synthetic-only execution remain unchanged.

Validation: `rtk proxy python -m unittest tests.test_serving_lifecycle -v` passed all 16 tests in 4.933s. Independently verified final freeze digest and 53-file membership; real dirty-checkout rejection confirmed. No source edits, ML/GPU execution, model downloads or installs. CPU event/socket fixtures establish protocol behavior only; actual CUDA cancellation, allocator-budget exhaustion and remote qualification remain pending root-owned evidence.

panel: full — security delta ✓ fresh-context session-model judge; combined/product/parsimony, Jev/Graft, signing and actual remote evidence remain separate parent-owned gates.

## Final confirmed-reap cleanup delta

Reviewed final freeze `063dde2eee1635ce1a52e40748ef76c06fb63a2fedb8a69da3322e269261015c`. Exact handle retention, explicit unconfirmed reap/error/PID, blocked restart, deferred buffered stream closure and later successful retry repair the false-reap finding. The focused double-timeout regression passed (one test, 0.205s), and the updated freeze independently verifies.

**must-fix (P2), `hala/serving/lifecycle.py:193`: retained ownership is still treated as an available worker.** `_exchange()` checks only `self.process is None`. After cleanup times out twice, that handle deliberately remains non-None, so a later request can reuse its private streams and consume an already queued response from the failed/cancelled request. CPU-only reproduction with a mock owned process: double-timeout `close()` left `reap.confirmed=false`; inserting a prior `{'output':'stale-output'}` in the message queue made the next ordinary engine call return `stale-output`. This violates fail-closed failure handling and the explicit fresh-worker restart contract. Preserve the handle for reap retries but mark the worker unavailable before cleanup begins; reject further exchanges until a confirmed reap and explicit successful start. Add a regression asserting no stdin write or queue output acceptance while the retained child is failed/unreaped.

No real subprocess, GPU, model download or actual qualification was involved in this reproduction. All actual CUDA claims remain pending.

## Final cycle 3: unavailable retained worker

Reviewed final freeze `4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a`; independent freeze verification passed. The stale-output P2 above is resolved: `usable` begins false, changes to true only after the fresh child's exact ready handshake, and becomes false before fault handling or cleanup. `_exchange()` rejects an unusable retained handle before any write or queue read, preserving the owned handle for explicit reap retries. Successful reap alone does not restore availability; a new successful start is required.

Validation: the stale-queue and double-reap regression tests both passed (2 tests, 0.809s), including unchanged stale queue, no stdin write, blocked unreaped restart, later successful reap, continued rejection until fresh start, and successful fresh-child inference. No remaining must-fix security findings in this narrow delta. No GPU/model work was performed; actual CUDA/allocator/cancellation qualification remains pending.

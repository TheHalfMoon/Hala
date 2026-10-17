# H23 consolidated review repair handoff

Worktree `work/Hala-lifecycle`, branch `ps/h23-serving-lifecycle`, exact base `733fc7e11cfc147a463d388ee65559acd1fa847b`. All changes remain uncommitted for root independent delta review. No existing H19/H21/H22 code changed.

Addressed source findings:

- Font/hash/Pillow/RAQM preflight now precedes the initial `WorkerEngine.start`, every explicit restart and child ML imports/download. Missing/unpinned font, missing RAQM and incorrect Pillow reject before start.
- The prospective freeze now covers the complete 53-file first-party Python superset plus config/tests. It explicitly includes `evaluation/__init__.py` and `evaluation/heldout_fixtures.py`; verification rejects omissions, additions and byte tampering. Both parent spawn and child load require an externally supplied exact 40-character source commit and fully clean Git checkout, including untracked files.
- Active CUDA cancellation has a genuine evidence route: TorchDispatchMode wraps real active-model CUDA matrix operations only (`mm`, `addmm`, `bmm`), records an event after enqueue, and keeps bounded 128-event history. A private inherited socketpair reaches an independent child control thread. Parent timestamps the shared transport Event.set, selects/query-acks an event enqueued before that receipt with a 250 ms absolute control deadline, records the pending/completed result, then terminates/reaps its exact owned child. No toy CUDA loop, public fault API or pre-enqueue/CPU marker is used as proof.
- Actual overlap qualification stays false until root independently inspects a real remote acknowledgment: event recorded before precise event_set cancellation receipt, still pending at later child receipt, real model operation identity, cancelled fault, no contradictory generation-return evidence and successful child reap. CPU fake event/control tests never qualify actual CUDA.
- Atomic result progress persists before initial load, each task request, resource/fault probe and reload, then after outcomes. Failures retain raw wire traces, cancellation acknowledgment, attempt bindings and reload error. Evidence is captured before risky cleanup; persistence failure cannot skip owned-child/server cleanup.

Regression evidence: four new tests initially failed against missing repair seams. Scoped suite now 16 tests passes, including bad-font-before-start, restart font/RAQM/version gate, dependency tampering and dirty source identity, bounded private pending-event ack selecting work before receipt, exact transport cancellation timestamp and partial progress surviving reload failure. Final full suite `rtk proxy python -m unittest discover -s tests -q`: 253 tests in 27.987s, OK with one skipped. Compileall passed. No GPU, model download or package install performed.

Final prospective freeze SHA256: `20b57ebb7e66b59f480cb89cb7fd363e671d212d48249ec99356856bc16730ae`. This supersedes all previous H23 hashes. Exact operator CLI and its new required `--source-commit` argument are in `experiments/serving-lifecycle/README.md`. Outputs must be outside the clean source checkout.

Root-owned remaining gates: independent combined/security delta review, genuine Jev/Graft, signing/commit and actual free remote execution. Actual event overlap, allocator-budget exhaustion, restart recovery, task quality, native integration and production remain unqualified until their actual evidence is independently checked.

## Second review repair: confirmed child reap

Addressed the sole must-fix in `work/h23-combined-delta-review.md`. `close()` now retains its exact owned Popen handle until bounded wait returns an actual exit code; terminate/wait and kill/wait timeouts record `reap.confirmed=false`, owned PID and error, and remain retryable. `start()` rejects an existing owned handle without attempting cleanup or spawning a replacement. Fault and final-result evidence now uses explicit confirmed reap, rather than a cleared-handle inference. Buffered pipe closure is deferred while unreaped to avoid blocking behind active reader/writer locks; private control closes and reader join remains bounded. A later successful close retry closes retained streams and confirms return code/PID before dropping ownership.

The CPU double-timeout regression failed before repair (owned handle lost), then passed with timeout bounds `[2,2]`, blocked restart/no new spawn, retained handle and later successful retry. Scoped suite: 17 tests passed. Final full suite: 254 tests in 28.066s, OK with one skipped. Compileall passed. No additional runtime scope, model/GPU work, existing H19/H21/H22 edits, commit or publication.

Current final freeze superseding all prior H23 hashes: `063dde2eee1635ce1a52e40748ef76c06fb63a2fedb8a69da3322e269261015c`. Actual qualification remains false. Root independent delta re-review remains required.

## Third/final review repair: failed ownership is not usable inference

Addressed the adjacent security P2: a retained unreaped process could previously consume stale queued model output. Added a separate usability state: false initially and before close/model/cleanup-fault handling; true only after an explicit new worker start passes the exact provenance handshake. Exchanges reject unusable ownership before writes, queue consumption or cleanup retry, preserving prior fault/reap evidence. Successful reap alone cannot enable inference; fresh explicit start is required.

The CPU regression failed before the fix, then passed with a deliberately queued stale output after double timeout: retained owned handle stays inaccessible, no pipe write/queue consumption/extra wait occurs, reap alone stays unavailable, and only new successful start recovers. Scoped suite: 18 tests passed. Final full suite: 255 tests in 25.905s, OK with one skipped. Compileall passed. No GPU/model/native work, existing H19/H21/H22 source edits, commit or publication.

Current final freeze superseding all prior H23 hashes: `4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a`. All actual qualification remains false. This reaches the three-cycle repair cap: final root independent delta review is required; any remaining must-fix parks the phase rather than further unattended cycling.
